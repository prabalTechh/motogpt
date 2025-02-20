import { DataAPIClient } from "@datastax/astra-db-ts";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import "dotenv/config";
import { pipeline } from "@huggingface/transformers";

type SimilarityMatrix = "euclidean" | "dot_product" | "cosine";

// Environment variables validation
const {
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_TOKEN,
} = process.env;

if (!ASTRA_DB_TOKEN || !ASTRA_DB_API_ENDPOINT || !ASTRA_DB_COLLECTION) {
  throw new Error("Missing required environment variables");
}

const motogptData = [
  "https://en.wikipedia.org/wiki/2025_MotoGP_World_Championship",
  "https://en.wikipedia.org/wiki/List_of_Grand_Prix_motorcycle_racing_winners",
];

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 100,
});

const client = new DataAPIClient(ASTRA_DB_TOKEN);
const db = client.db(ASTRA_DB_API_ENDPOINT, {
  namespace: ASTRA_DB_NAMESPACE || "default_namespace",
});

// Initialize embedder globally with proper async handling
let embedder: any;
const initializeEmbedder = async () => {
  console.log("Initializing embedder...");
  embedder = await pipeline("feature-extraction", "sentence-transformers/all-MiniLM-L6-v2");
  console.log("Embedder initialized");
};

const createCollection = async (similarityMatrix: SimilarityMatrix = "dot_product") => {
  try {
    // Check if collection exists and drop it to ensure correct dimension
    const collections = await db.listCollections();
    //@ts-ignore
    if (collections.includes(ASTRA_DB_COLLECTION)) {
      console.log("Dropping existing collection...");
      await db.dropCollection(ASTRA_DB_COLLECTION);
    }

    const res = await db.createCollection(ASTRA_DB_COLLECTION, {
      vector: {
        dimension: 384, // Matches all-MiniLM-L6-v2 output
        metric: similarityMatrix,
      },
    });
    console.log("Collection created:", res);
    return res;
  } catch (error) {
    console.error("Error creating collection:", error);
    throw error;
  }
};

const scrapePage = async (url: string): Promise<string> => {
  try {
    const loader = new PuppeteerWebBaseLoader(url, {
      launchOptions: { headless: true },
      gotoOptions: { waitUntil: "domcontentloaded" },
      evaluate: async (page, browser) => {
        const result = await page.evaluate(() => document.body.innerText);
        await browser.close();
        return result;
      },
    });
    const content = await loader.scrape();
    return content?.replace(/[^\w\s.,!?]/g, " ") || "";
  } catch (error) {
    console.error(`Error scraping ${url}:`, error);
    return "";
  }
};

const loadSample = async () => {
  try {
    const collection = await db.collection(ASTRA_DB_COLLECTION);

    for (const url of motogptData) {
      console.log(`Scraping ${url}...`);
      const content = await scrapePage(url);

      if (!content) {
        console.warn(`No content retrieved from ${url}`);
        continue;
      }

      const chunks = await splitter.splitText(content);
      console.log(`Processing ${chunks.length} chunks from ${url}`);

      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        try {
          const embedding = await embedder(chunk, { pooling: "mean", normalize: true });
          const vector = Array.from(embedding.data);

          // Validate vector
          if (vector.length !== 384) {
            throw new Error(`Vector dimension mismatch: expected 384, got ${vector.length}`);
          }
          if (!vector.every((v) => typeof v === "number" && !isNaN(v))) {
            throw new Error("Vector contains invalid values");
          }

          const res = await collection.insertOne({
            $vector: vector,
            text: chunk,
          });
          console.log(`Chunk ${i + 1} inserted:`, res);
        } catch (error:any){
          console.error(`Error inserting chunk ${i + 1}:`, error);
          if (error.response) {
            console.error("Server response:", error.response.data);
          }
        }
      }
    }
  } catch (error) {
    console.error("Error in loadSample:", error);
    throw error;
  }
};

const main = async () => {
  try {
    await initializeEmbedder(); // Ensure embedder is ready
    await createCollection(); // Uncommented to ensure collection exists
    await loadSample();
    console.log("Process completed successfully");
  } catch (error) {
    console.error("Main process failed:", error);
  }
};

main();