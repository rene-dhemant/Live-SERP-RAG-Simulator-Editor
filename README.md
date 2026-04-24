# Live SERP RAG Simulator & Editor

A Node.js-based diagnostic tool that reverse-engineers the Retrieval-Augmented Generation (RAG) pipelines used by modern generative answer systems (e.g., Perplexity, Google AI Overviews). 

This application simulates the exact deterministic retrieval and probabilistic generation phases of an AI search engine, allowing technical SEOs and content architects to measure and optimize their "Share of Context" (SoC) against live SERP competitors.

![image](https://dhemant.consulting/wp-content/uploads/rag-sim-dcg.jpg)

## Technical Architecture & Pipeline

The tool operates on an Express.js backend and executes a multi-stage RAG pipeline using a two-tier filtering system (Bi-Encoder and Cross-Encoder):

1. **Concurrent Data Ingestion:** The server receives a target query and URL. It concurrently fetches the target URL's DOM via ScraperAPI and the Top 10 organic SERP winners via the Tavily Search API.

2. **Extraction & Chunking:** Uses `cheerio` to strip DOM noise (scripts, styling, navigation). The remaining core content is parsed into structured Markdown and divided into semantic chunks (~800 characters) respecting heading boundaries.

3. **Semantic Vector Search (Bi-Encoder):** All chunks are vectorized via batch processing using the `gemini-embedding-001` model. The server calculates the exact Cosine Similarity between the User Prompt vector and the Chunk vectors to establish an initial Top 20 Evaluation Pool.

4. **Generative Reranking (LLM-as-a-Judge / Cross-Encoder):** The Top 20 pool is passed to `gemini-2.5-pro` (operating at Temperature 0.0 for deterministic output). The LLM blindly scores each chunk from 0 to 3 based on structural metrics: Factual Density, Directness (BLUF), and Completeness.

5. **Context Window Assembly & Synthesis:** The system sorts the final Top 5 chunks based primarily on the LLM Score and secondarily on Cosine Similarity. This forms the final Context Window. The LLM then synthesizes a simulated AI Overview with inline citations and generates a data-dense rewrite of the user's content to compete for the #1 citation slot.

![image](https://dhemant.consulting/wp-content/uploads/rag-pipeline-overview.png)

## Prerequisites

* **Node.js** (v18.0.0 or higher recommended)
* **API Keys** for the following services:
  * [Google Gemini API](https://aistudio.google.com/) (For embeddings and LLM reranking/synthesis)
  * [Tavily API](https://tavily.com/) (For SERP retrieval and text scraping)
  * [ScraperAPI](https://www.scraperapi.com/) (For bypassing bot protection on the target URL)

## Installation Guide

**1. Clone the repository**

Install dependencies

The application requires Express, CORS, Axios, Cheerio, and Dotenv.
```bash
npm install express cors axios cheerio dotenv
```

**2. Configure Environment Variables**
Edit .env file in the root directory of the project and add your API keys.
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
SCRAPER_API_KEY=your_scraperapi_key_here
TAVILY_API_KEY=your_tavily_api_key_here
PORT=3000
```

**3. Start the server**
```bash
node server.js
```

**4. Access the Application**
Open your web browser and navigate to:
```bash
http://localhost:3000
```

_(Note: For production deployment, it is highly recommended to route the application through a secure reverse proxy such as Cloudflare Tunnels or Nginx/OpenLiteSpeed to manage SSL and port forwarding.)_


![My Logo](https://dhemant.consulting/wp-content/uploads/dhemant-seo-consultancy-services-333.svg)
