import { DataAPIClient } from "@datastax/astra-db-ts";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import OpenAI from "openai";


import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";


import "dotenv/config";
import { headers } from "next/headers";
type similarityMatrix = "euclidean" | "dot_product" | "cosine"



const {
    ASTRA_DB_NAMESPACE,
    ASTRA_DB_COLLECTION,
    ASTRA_DB_API_ENDPOINT,
    ASTRA_DB_TOKEN,
    OPENAI_API_KEY
} = process.env;


const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

const motogptData = [
    "https://en.wikipedia.org/wiki/2025_MotoGP_World_Championship",
    "https://en.wikipedia.org/wiki/List_of_Grand_Prix_motorcycle_racing_winners",
    // "https://www.crash.net/motogp/feature/1046664/1/2025-motogp-rider-line-who-confirmed-and-rumoured-2025-grid",
    // "https://www.motogp.com/en",
    // "https://www.giantbomb.com/motogp/3025-542/locations/",
    // "https://www.viagogo.com/in/Sports-Tickets/Motorsports/MotoGP?=&PCID=PSINADWHOME42768638023505&MetroRegionID=&psc=&ps=&ps_p=0&ps_c=22002985832&ps_ag=171844435197&ps_tg=kwd-980231799&ps_ad=724830813485&ps_adp=&ps_fi=&ps_li=&ps_lp=1007765&ps_n=g&ps_d=c&ps_ex=&pscpag=&gad_source=1&gclid=Cj0KCQiAwtu9BhC8ARIsAI9JHalKs8AMjo-Yjnvi4czwpaTO7dYv7kfgUL1W031txO4yjtht4X7Mtw8aAhT4EALw_wcB"
]


const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 512,
    chunkOverlap: 100
})


const client = new DataAPIClient(ASTRA_DB_TOKEN);

const db = client.db(ASTRA_DB_API_ENDPOINT as string, { namespace: ASTRA_DB_NAMESPACE });


const createCollection = async (similarityMatrix: similarityMatrix = "dot_product") => {

    const res = await db.createCollection(ASTRA_DB_COLLECTION as string,
        {
            vector: {
                dimension: 1536,
                metric: similarityMatrix,
            }
        }
    );
    console.log(res);
}


const loadSample = async () => {
    const collection = await db.collection(ASTRA_DB_COLLECTION as string);
    for await (const url of motogptData) {
        const content = await scrapePage(url);
        const chunks = await splitter.splitText(content);
        for await (const chunk of chunks) {
            const embedding = await openai.embeddings.create({
                model: "text-embedding-3-small",
                input: chunk,
                encoding_format: "float",
            });

            const vector = embedding.data[0].embedding

            const res = await collection.insertOne({
                $vector : vector,
                text : chunk
            });
            console.log(res);
        }
    }
}


const scrapePage = async(url:string) => {
    const loader = new PuppeteerWebBaseLoader(url,{
        launchOptions : {headless:true},
        gotoOptions : {waitUntil:"domcontentloaded"},
        evaluate : async(page: { evaluate: (arg0: () => string) => any; }, browser: { close: () => any; }) =>{
            const result = await page.evaluate(()=>document.body.innerHTML)
            await browser.close()
            return result;
        }
    });
    return (await loader.scrape())?.replace(/<[^>]*>?/gm, '')
}



createCollection().then(()=> loadSample());

