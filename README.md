# Live SERP RAG Simulator & Editor

A Node.js-based diagnostic tool that reverse-engineers the Retrieval-Augmented Generation (RAG) pipelines used by modern generative answer systems (e.g., Perplexity, Google AI Overviews). 

This application simulates the exact deterministic retrieval and probabilistic generation phases of an AI search engine, allowing technical SEOs and content architects to measure and optimize their "Share of Context" (SoC) against live SERP competitors using advanced techniques like HyDE and Parent-Child chunking.

![image](https://dhemant.consulting/wp-content/uploads/rag-sim-dcg.jpg)

## Technical Architecture & Pipeline

The tool operates on an Express.js backend and executes a multi-stage RAG pipeline using a two-tier filtering system (Bi-Encoder and Cross-Encoder):

1. **Concurrent Data Ingestion:** The server receives a target query and URL. It concurrently fetches the target URL's DOM via ScraperAPI and the Top 10 organic SERP winners via the Tavily Search API.

2. **Parent-Child Chunking:** Content is parsed into structured Markdown and divided into two tiers: `Parent` chunks (~800 characters) for LLM context, and `Child` chunks (~200 characters) for high-precision mathematical embedding.

3. **HyDE (Hypothetical Document Embeddings):** The pipeline prompts a fast LLM (`gemini-2.5-flash`) to generate a hypothetical "perfect" answer to the user's query. This document is vectorized instead of the raw query to eliminate vocabulary mismatch.

4. **Semantic Vector Search (Bi-Encoder):** The `Child` chunks are vectorized via `gemini-embedding-001`. The server calculates the Cosine Similarity between the HyDE vector and the Child vectors.

5. **Auto-Merging:** The highest-scoring Child chunks pass their similarity scores back to their respective Parent chunks. The top 20 Parents form the Evaluation Pool.

6. **Generative Reranking (LLM-as-a-Judge / Cross-Encoder):** The Top 20 pool is passed to `gemini-2.5-pro` (Temperature 0.0). The LLM blindly scores each Parent chunk from 0 to 3 based on Factual Density, Directness (BLUF), and Completeness.

7. **Advanced Analytics & Synthesis:**
    * **Entity Gap Analysis:** Extracts Named Entities present in competitor chunks but missing from the user's chunk.
    * **Information Gain Scoring:** Evaluates if the user's chunk provides net-new data compared to the SERP consensus.
    * **Stochastic Simulation:** Fires 5 concurrent LLM generation requests (Temperature 0.7) using the final Context Window to calculate the exact Citation Probability, accounting for AI variance.
    * **AIO Rewrite:** Generates a data-dense rewrite of the user's content to compete for the #1 citation slot.

![image](https://dhemant.consulting/wp-content/uploads/rag-pipeline-simulator.png)

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

![image](https://dhemant.consulting/wp-content/uploads/ragsim-gui.png)

![My Logo](https://dhemant.consulting/wp-content/uploads/dhemant-seo-consultancy-services-333.svg)
