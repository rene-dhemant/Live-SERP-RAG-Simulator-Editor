require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const cheerio = require('cheerio');
const crypto = require('crypto');
const path = require('path');
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ==========================================
// CONFIGURATION
// ==========================================
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const SCRAPER_API_KEY = process.env.SCRAPER_API_KEY;
const TAVILY_API_KEY = process.env.TAVILY_API_KEY;

// ==========================================
// CORE PIPELINE ROUTE
// ==========================================
app.post('/api/analyze', async (req, res) => {
    const { query, my_url } = req.body;

    if (!query || !my_url) {
        return res.status(400).json({ error: "Please provide the Query and Your URL." });
    }

    try {
        const [myRawHtml, tavilyResults] = await Promise.all([
            fetchUrlWithScraperApi(my_url),
            fetchTavilySerp(query)
        ]);

        if (!myRawHtml) throw new Error("Failed to fetch Your URL. Check your ScraperAPI key.");
        if (!tavilyResults) throw new Error("Failed to fetch SERP data from Tavily.");

        const myData = extractPageData(myRawHtml);
        const myMarkdown = basicHtmlToMarkdown(myData.content);

        if (!myMarkdown) throw new Error("Successfully fetched your page, but could not extract valid text content.");

        const results = await runMasterRagPipeline(query, my_url, myMarkdown, myData.title, tavilyResults);
        
        if (results.error) throw new Error(results.error);

        const htmlOutput = renderResultsHtml(results);
        res.json({ success: true, html: htmlOutput });

    } catch (error) {
        console.error("Pipeline Error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

// ==========================================
// EXTRACTION & FETCH FUNCTIONS
// ==========================================
async function fetchUrlWithScraperApi(url) {
    try {
        const response = await axios.get(`http://api.scraperapi.com/?api_key=${SCRAPER_API_KEY}&url=${encodeURIComponent(url)}`, { timeout: 60000 });
        return response.data;
    } catch (e) { return false; }
}

async function fetchTavilySerp(query) {
    try {
        const response = await axios.post('https://api.tavily.com/search', {
            api_key: TAVILY_API_KEY,
            query: query,
            search_depth: "advanced",
            max_results: 10
        }, { timeout: 45000 });
        return response.data.results || false;
    } catch (e) { return false; }
}

function extractPageData(html) {
    if (!html) return { title: 'Unknown', content: '' };
    const $ = cheerio.load(html);
    const title = $('title').text().trim() || 'Unknown Document';
    
    // Radical noise stripping
    $('script, style, nav, header, footer, aside, noscript, iframe, svg, form, button, .footer, #footer, .cookie-banner').remove();
    
    let contentHtml = "";
    
    // Intelligent container targeting
    const selectors = ['main', 'article', '#main', '#content', '.main-content', '.post-content', '.content'];
    
    for (const selector of selectors) {
        const node = $(selector).first();
        if (node.length > 0) {
            // Validate the container has substantial text content
            const textLength = node.text().trim().length;
            if (textLength > 300) {
                contentHtml = node.html();
                break;
            }
        }
    }
    
    // Fallback to body if structural containers failed
    if (!contentHtml) {
        contentHtml = $('body').html() || html;
    }
    
    return { title, content: contentHtml };
}

function basicHtmlToMarkdown(html) {
    let md = html.replace(/<h1[^>]*>(.*?)<\/h1>/gis, "\n\n# $1\n\n");
    md = md.replace(/<h2[^>]*>(.*?)<\/h2>/gis, "\n\n## $1\n\n");
    md = md.replace(/<h3[^>]*>(.*?)<\/h3>/gis, "\n\n### $1\n\n");
    md = md.replace(/<p[^>]*>(.*?)<\/p>/gis, "\n\n$1\n\n");
    md = md.replace(/<br[^>]*>/gi, "\n");
    md = md.replace(/<strong[^>]*>(.*?)<\/strong>/gis, "**$1**");
    md = md.replace(/<li[^>]*>(.*?)<\/li>/gis, "\n* $1");
    md = md.replace(/(<([^>]+)>)/gi, "");
    
    // Aggressive Whitespace Normalization
    md = md.replace(/^[ \t]+$/gm, ""); // Remove lines containing only spaces
    md = md.replace(/[ \t]+/g, " ");   // Collapse multiple inline spaces
    md = md.replace(/\n{3,}/g, "\n\n"); // Collapse 3+ newlines into exactly 2
    
    return md.trim();
}

// ==========================================
// PIPELINE LOGIC & GOOGLE AI CALLS
// ==========================================
async function runMasterRagPipeline(query, myUrl, myContent, myTitle, tavilyResults) {
    
    // 1. Build Semantic Parent/Child Chunks for Your Site
    const myData = await buildSmartChunks(myContent, 'you', myTitle, myUrl);
    let allParents = [...myData.parents];
    let allChildren = [...myData.children];
    
    // 2. Build Semantic Parent/Child Chunks for SERP Competitors
    const serpClassifications = await ragClassifySerpUrls(tavilyResults);
    let tavilyInfo = [];
    const myDomain = new URL(myUrl).hostname;

    for (const result of tavilyResults) {
        if (result.url.includes(myDomain)) continue;
        const pageType = serpClassifications[result.url] || 'News/Article';
        tavilyInfo.push({ url: result.url, title: result.title, type: pageType });
        
        const compData = await buildSmartChunks(result.content, 'comp', result.title, result.url);
        allParents = allParents.concat(compData.parents);
        allChildren = allChildren.concat(compData.children);
    }

    if (allParents.length === 0) return { error: "No valid text chunks found." };

    // 3. HyDE: Generate Fake Document & Embed It
    const hydeDocument = await ragGenerateHyDE(query);
    const hydeEmbedding = await ragGetEmbedding(hydeDocument);
    if (!hydeEmbedding) return { error: "Failed to generate HyDE embedding." };

    // 4. Batch Embed Only the Small CHILD Chunks
    const textsToEmbed = allChildren.map(c => c.text);
    const batchEmbeddings = await ragBatchGetEmbeddings(textsToEmbed);
    if (!batchEmbeddings || batchEmbeddings.length !== textsToEmbed.length) {
        return { error: "Failed to generate batch embeddings." };
    }

    // 5. Calculate Semantic Match on Children
    allChildren.forEach((child, index) => {
        child.similarity = cosineSimilarity(hydeEmbedding, batchEmbeddings[index]);
    });

    // 6. Sort Children and Map back to Parents (Auto-Merging)
    allChildren.sort((a, b) => b.similarity - a.similarity);
    
    let mappedParents = [];
    let seenParentIds = new Set();

    for (const child of allChildren) {
        if (!seenParentIds.has(child.parentId)) {
            seenParentIds.add(child.parentId);
            const parent = allParents.find(p => p.id === child.parentId);
            if (parent) {
                parent.similarity = child.similarity; 
                mappedParents.push(parent);
            }
        }
    }

    // Filter Top 20 Parents for LLM Judge
    let myScored = mappedParents.filter(p => p.source === 'you').slice(0, 5);
    let compScored = mappedParents.filter(p => p.source === 'comp').slice(0, 15);
    const top20Retrieved = [...myScored, ...compScored];

    // 7. Generative Reranking (Cross-Encoder)
    const gradedResults = await ragBatchLlmRerank(query, top20Retrieved);
    if (!gradedResults) return { error: "Failed to rerank chunks." };

    top20Retrieved.forEach(chunk => {
        const evalData = gradedResults[chunk.id];
        chunk.llm_score = evalData ? parseInt(evalData.score) : 0;
        chunk.llm_reason = evalData ? evalData.rationale : 'Failed to process.';
        chunk.readiness = evalData ? evalData.readiness : { density: 'Low', directness: 'Low', completeness: 'Low' };
    });

    // 8. Final Sort for Context Window Assembly
    top20Retrieved.sort((a, b) => {
        if (b.llm_score === a.llm_score) return b.similarity - a.similarity;
        return b.llm_score - a.llm_score;
    });

    const finalTop5 = top20Retrieved.slice(0, 5);
    const myTop5Count = finalTop5.filter(c => c.source === 'you').length;
    const compTop5Count = finalTop5.filter(c => c.source === 'comp').length;
    
    let citationWinnerSource = 'none';
    if (finalTop5.length > 0 && finalTop5[0].llm_score >= 2) citationWinnerSource = finalTop5[0].source;

    let bestYou = null, bestComp = null;
    top20Retrieved.forEach(chunk => {
        if (chunk.source === 'you' && (!bestYou || chunk.llm_score > bestYou.llm_score)) bestYou = chunk;
        if (chunk.source === 'comp' && !bestComp) bestComp = chunk;
    });

    // 9. Analytics & Optimization Phase
    const compTextsInTop5 = finalTop5.filter(c => c.source === 'comp').map(c => c.text);
    
    const [simulationsData, optimizedRewrite, entityGap, infoGain] = await Promise.all([
        ragGenerateSynthesisSimulations(query, finalTop5, myUrl),
        (bestComp ? ragGenerateOptimizedChunk(query, bestYou ? bestYou.text : '', bestComp.text) : Promise.resolve('')),
        (bestYou && compTextsInTop5.length > 0 ? ragAnalyzeEntityGap(bestYou.text, compTextsInTop5) : Promise.resolve([])),
        (bestYou && compTextsInTop5.length > 0 ? ragScoreInformationGain(bestYou.text, compTextsInTop5) : Promise.resolve(null))
    ]);

    return {
        tavily_info: tavilyInfo, my_top_5_count: myTop5Count, comp_top_5_count: compTop5Count,
        citation_winner_source: citationWinnerSource, best_you: bestYou, best_comp: bestComp,
        ai_overview: simulationsData.overview, citation_probability: simulationsData.probability,
        optimized_rewrite: optimizedRewrite, top_5_chunks: finalTop5, 
        entity_gap: entityGap, info_gain: infoGain,
        my_extracted_text: myContent // Included for the Debug Accordion
    };
}

// LANGCHAIN SEMANTIC PARENT-CHILD CHUNKING LOGIC
async function buildSmartChunks(content, sourceTag, title, url) {
    let parents = [];
    let children = [];
    const uniquePrefix = sourceTag + '_' + crypto.createHash('md5').update(url).digest('hex').substring(0, 5);
    const domain = new URL(url).hostname;

    const parentSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 800,
        chunkOverlap: 100, 
        separators: ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    });
    
    const childSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 200,
        chunkOverlap: 40, 
        separators: [". ", "? ", "! ", "\n", " ", ""] 
    });

    const parentTexts = await parentSplitter.splitText(content);

    for (let i = 0; i < parentTexts.length; i++) {
        const pId = `${uniquePrefix}_p${i}`;
        const parentFormattedText = `Source: ${domain} | Title: ${title}\n---\n${parentTexts[i]}`;
        
        parents.push({ 
            id: pId, 
            source: sourceTag, 
            text: parentFormattedText, 
            url: url 
        });

        const childTexts = await childSplitter.splitText(parentTexts[i]);
        
        for (let j = 0; j < childTexts.length; j++) {
            children.push({ 
                id: `${pId}_c${j}`, 
                parentId: pId, 
                source: sourceTag, 
                text: childTexts[j] 
            });
        }
    }

    return { parents, children };
}

function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0, normA = 0, normB = 0;
    for (let i = 0; i < Math.min(vecA.length, vecB.length); i++) {
        dotProduct += vecA[i] * vecB[i]; normA += vecA[i] ** 2; normB += vecB[i] ** 2;
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function ragGenerateHyDE(query) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GOOGLE_API_KEY}`;
    const prompt = `Write a hypothetical, highly factual, and concise 1-paragraph answer to the following query. Do not use filler words. Just output the raw hypothetical answer.\nQuery: ${query}`;
    
    try {
        const response = await axios.post(url, { contents: [{ parts: [{ text: prompt }] }], generationConfig: { temperature: 0.2 } }, { timeout: 45000 });
        return response.data.candidates[0].content.parts[0].text.trim();
    } catch (e) { 
        return query; 
    }
}

async function ragGetEmbedding(text) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=${GOOGLE_API_KEY}`;
    try {
        const response = await axios.post(url, { model: 'models/gemini-embedding-001', content: { parts: [{ text }] } });
        return response.data.embedding.values;
    } catch (e) { return false; }
}

async function ragBatchGetEmbeddings(chunksText) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key=${GOOGLE_API_KEY}`;
    let allEmbeddings = [];
    const chunkSize = 90;
    for (let i = 0; i < chunksText.length; i += chunkSize) {
        const batch = chunksText.slice(i, i + chunkSize);
        const requests = batch.map(text => ({ model: 'models/gemini-embedding-001', content: { parts: [{ text }] } }));
        try {
            const response = await axios.post(url, { requests }, { timeout: 60000 });
            if (!response.data.embeddings) return false;
            response.data.embeddings.forEach(emb => allEmbeddings.push(emb.values));
        } catch (e) { return false; }
    }
    return allEmbeddings;
}

async function ragClassifySerpUrls(tavilyResults) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${GOOGLE_API_KEY}`;
    let prompt = "Classify the following web pages based on their Title and URL into exactly ONE of these categories: [Aggregator/Listicle, Vendor/Product Page, Comparison/Review, Market Overview, Forum/UGC, News/Article, Other].\n\n";
    tavilyResults.forEach(res => prompt += `URL: ${res.url}\nTitle: ${res.title}\n\n`);
    prompt += "Output ONLY a valid JSON object where keys are URLs and values are categories.";

    try {
        const response = await axios.post(url, { contents: [{ parts: [{ text: prompt }] }], generationConfig: { temperature: 0.0, response_mime_type: 'application/json' } }, { timeout: 60000 });
        return JSON.parse(response.data.candidates[0].content.parts[0].text);
    } catch (e) { return {}; }
}

async function ragBatchLlmRerank(query, chunks) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${GOOGLE_API_KEY}`;
    let prompt = `You are a strict AI Optimization Judge. Evaluate these text chunks based on how well they answer the User Query.\n\nUser Query: ${query}\n\n`;
    chunks.forEach(c => prompt += `Chunk ID [${c.id}]:\n${c.text}\n\n`);
    prompt += "Output ONLY a valid JSON object where keys are Chunk IDs. The value must be an object with: 'score' (0-3), 'rationale' (1 sentence), 'readiness' (density, directness, completeness levels).";

    try {
        const response = await axios.post(url, { contents: [{ parts: [{ text: prompt }] }], generationConfig: { temperature: 0.0, response_mime_type: 'application/json' } }, { timeout: 90000 });
        return JSON.parse(response.data.candidates[0].content.parts[0].text);
    } catch (e) { return false; }
}

async function ragGenerateSynthesisSimulations(query, topChunks, myUrl) {
    if (!topChunks.length) return { overview: "Not enough context found.", probability: 0 };
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${GOOGLE_API_KEY}`;
    let prompt = `You are an AI Search Engine. Synthesize a final answer using ONLY the provided chunks.\nUser Query: ${query}\n\n`;
    topChunks.forEach((c, i) => { prompt += `--- Chunk ${i + 1} ---\nURL: ${c.url}\nText: ${c.text}\n\n`; });
    prompt += `RULES: 1. Write 1-2 paragraphs. 2. Format in clean HTML (<p>, <br>). 3. Bold brand names. 4. You MUST cite sources inline at the end of sentences using: <a href="THE_URL_FROM_THE_CHUNK" class="ai-citation" target="_blank">[1]</a> (replace 1 with the chunk number).`;

    const requests = Array(5).fill().map(() => 
        axios.post(url, { contents: [{ parts: [{ text: prompt }] }], generationConfig: { temperature: 0.7 } }, { timeout: 90000 })
    );

    try {
        const responses = await Promise.all(requests);
        let overviews = responses.map(res => res.data.candidates[0].content.parts[0].text.trim());
        
        let citationHits = 0;
        overviews.forEach(text => {
            if (text.includes(myUrl)) citationHits++;
        });

        return { overview: overviews[0], probability: (citationHits / 5) * 100 };
    } catch (e) { return { overview: 'Could not generate simulations.', probability: 0 }; }
}

async function ragGenerateOptimizedChunk(query, userText, compText) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${GOOGLE_API_KEY}`;
    let prompt = `You are an expert AI Search Optimizer. Rewrite the User's text to become the #1 cited answer without promoting competitors.\nUser Query: ${query}\nCompetitor's Winning Text: ${compText}\nUser's Current Text: ${userText}\nRULES: No competitors, emulate data structure, objective tone, output ONLY Markdown.`;

    try {
        const response = await axios.post(url, { contents: [{ parts: [{ text: prompt }] }], generationConfig: { temperature: 0.1 } }, { timeout: 90000 });
        return response.data.candidates[0].content.parts[0].text.trim();
    } catch (e) { return 'Failed to generate optimized text.'; }
}

async function ragAnalyzeEntityGap(myText, compTexts) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${GOOGLE_API_KEY}`;
    const prompt = `Extract the top 5 most important Named Entities (specific tools, brand names, statistics, methodologies, concepts) present in the Competitor Texts that are MISSING from the User Text.
    User Text: ${myText}
    Competitor Texts: ${compTexts.join('\n---\n')}
    Output ONLY a valid JSON array of strings representing the missing entities. If none, output [].`;

    try {
        const response = await axios.post(url, { contents: [{ parts: [{ text: prompt }] }], generationConfig: { temperature: 0.0, response_mime_type: 'application/json' } }, { timeout: 60000 });
        return JSON.parse(response.data.candidates[0].content.parts[0].text);
    } catch (e) { return []; }
}

async function ragScoreInformationGain(myText, compTexts) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${GOOGLE_API_KEY}`;
    const prompt = `Analyze the User Text against the Competitor Texts. Does the User Text provide unique "Information Gain" (net-new valuable information, unique perspectives, or novel data) not found in the competitors?
    User Text: ${myText}
    Competitor Texts: ${compTexts.join('\n---\n')}
    Output ONLY a valid JSON object: {"score": "Low" | "Medium" | "High", "rationale": "1 short sentence explanation"}`;

    try {
        const response = await axios.post(url, { contents: [{ parts: [{ text: prompt }] }], generationConfig: { temperature: 0.0, response_mime_type: 'application/json' } }, { timeout: 60000 });
        return JSON.parse(response.data.candidates[0].content.parts[0].text);
    } catch (e) { return { score: "Low", rationale: "Failed to analyze." }; }
}

// ==========================================
// HTML TEMPLATE RENDERER
// ==========================================
function renderResultsHtml(results) {
    const tipSoC = "Share of Context (SoC) shows how many of the final Top 5 slots are occupied by your website vs competitors. Holding more slots mathematically increases your chance of controlling the final generative AI narrative.";
    const tipScore = "The AI Judge graded this text from 0 to 3 based on Factual Density, Directness, and Completeness.";
    const tipVector = "The mathematical proximity (Cosine Similarity) between the HyDE Document and the Child Text Chunk. Mapped to the Parent Chunk via Auto-Merging.";
    const tipWinner = "This chunk achieved the highest combined LLM Score and Semantic Match, making it the most likely text to be cited as the #1 source by an AI Search Engine.";
    const tipInfoGain = "Measures whether your text adds net-new valuable data or unique perspectives to the LLM, rather than just repeating competitor consensus.";
    const tipProbability = "We ran the AI synthesis engine 5 times consecutively to account for stochastic variance. This is the percentage of times your URL was successfully cited.";

    const mySoc = (results.my_top_5_count / 5) * 100;
    const compSoc = (results.comp_top_5_count / 5) * 100;
    
    let html = `<div class="rag-results"><h2>Deep Analysis Results</h2>`;
    
    if (results.tavily_info.length) {
        html += `<div class="serp-list"><h4>🔍 Live Organic Competitors (Top 10 Intent Analysis)</h4><ul>`;
        results.tavily_info.forEach(info => {
            const badgeClass = info.type.toLowerCase().replace(/[^a-z0-9]+/g, '-');
            html += `<li><span class="serp-badge ${badgeClass}">${info.type}</span><a href="${info.url}" target="_blank">${info.url}</a></li>`;
        });
        html += `</ul></div>`;
    }

    if (results.my_extracted_text) {
        html += `<details class="extraction-accordion">
                    <summary>🛠️ Debug: View Extracted Text from Your URL</summary>
                    <div class="extraction-content">${results.my_extracted_text.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</div>
                 </details>`;
    }

    html += `<div class="kpi-grid">
                <div class="kpi-card">
                    <div class="soc-tooltip-wrapper">
                        <h4 data-tip="${tipSoC}" class="kpi-title">Share of Context (Top 5 Chunks)</h4>
                    </div>
                    <div class="kpi-soc-labels"><span>You: ${results.my_top_5_count}</span><span>SERP: ${results.comp_top_5_count}</span></div>
                    <div class="soc-bar">
                        ${mySoc > 0 ? `<div class="soc-you" style="width: ${mySoc}%;"></div>` : ''}
                        ${compSoc > 0 ? `<div class="soc-comp" style="width: ${compSoc}%;"></div>` : ''}
                    </div>
                </div>
                <div class="kpi-card">
                    <h4 data-tip="${tipProbability}" class="kpi-title" style="border-bottom:1px dotted #94a3b8; display:inline-block; margin-bottom:10px;">Stochastic Citation Probability</h4>
                    <div class="probability-score ${results.citation_probability >= 60 ? 'high' : (results.citation_probability > 0 ? 'med' : 'low')}">${results.citation_probability}%</div>
                    <div class="probability-subtext">Based on 5 simulated generative iterations.</div>
                </div>
             </div>`;

    html += `<h3 style="margin-bottom:10px; color:#0f172a; margin-top: 30px;">Head-to-Head: Synthesis Readiness</h3>`;
    html += `<div class="rag-info-alert">💡 <strong>Pro Tip: High Semantic Match but Low LLM Score?</strong> Semantic Match gets you in the door; factual density wins the citation.</div>`;
    
    html += `<div class="vs-grid">`;
    if (results.best_you) {
        const isWinner = results.citation_winner_source === 'you';
        const semMatch = Math.round(results.best_you.similarity * 100);
        const r = results.best_you.readiness;
        const infoGain = results.info_gain;
        
        html += `<div class="vs-card ${isWinner ? 'winner-card' : ''}">
                    ${isWinner ? `<div class="vs-badge" data-tip="${tipWinner}">🏆 #1 Citation</div>` : ''}
                    <div class="vs-header"><div><span style="color:var(--win-green);">Your Best Chunk</span></div>
                    <div class="metric-group"><span class="vs-score" data-tip="${tipScore}">LLM Score: ${results.best_you.llm_score}/3</span><span class="vs-vector" data-tip="${tipVector}">🎯 Semantic Match: ${semMatch}%</span></div></div>
                    <div class="rag-ai-rationale">🤖 <strong>AI Rationale:</strong> ${results.best_you.llm_reason}</div>
                    <div class="synthesis-box"><h5>AI Readiness & Data Report</h5>
                        <div class="synth-metric"><span class="synth-label">Density:</span> <span class="synth-value ${r.density.toLowerCase()}">${r.density}</span></div>
                        <div class="synth-metric"><span class="synth-label">Directness:</span> <span class="synth-value ${r.directness.toLowerCase()}">${r.directness}</span></div>
                        <div class="synth-metric"><span class="synth-label">Completeness:</span> <span class="synth-value ${r.completeness.toLowerCase()}">${r.completeness}</span></div>
                        ${infoGain ? `<div class="synth-metric" style="margin-top:8px; padding-top:8px; border-top:1px dashed #e2e8f0;"><span class="synth-label" data-tip="${tipInfoGain}">Information Gain:</span> <span class="synth-value ${infoGain.score.toLowerCase()}">${infoGain.score}</span></div>
                        <div style="font-size: 11px; color: #64748b; line-height: 1.4; margin-top:4px;">${infoGain.rationale}</div>` : ''}
                    </div>`;
                    
        if (results.entity_gap && results.entity_gap.length > 0) {
            html += `<div class="entity-gap-box">
                        <strong style="color: #b91c1c; display:block; margin-bottom:5px;">⚠️ Entity Gap Warning</strong>
                        The SERP winners include these entities which are missing from your chunk: 
                        <div style="margin-top:5px;">${results.entity_gap.map(e => `<span class="entity-badge">${e}</span>`).join('')}</div>
                     </div>`;
        }

        html += `<span class="chunk-label">Extracted Text (Parent):</span><div class="vs-quote">${results.best_you.text}</div>
                 </div>`;
    }

    if (results.best_comp) {
        const isWinner = results.citation_winner_source === 'comp';
        const semMatch = Math.round(results.best_comp.similarity * 100);
        const r = results.best_comp.readiness;
        html += `<div class="vs-card ${isWinner ? 'winner-card' : ''}">
                    ${isWinner ? `<div class="vs-badge" data-tip="${tipWinner}">🏆 #1 Citation</div>` : ''}
                    <div class="vs-header"><div><span style="color:var(--lose-red);">SERP's Best Chunk</span></div>
                    <div class="metric-group"><span class="vs-score" data-tip="${tipScore}">LLM Score: ${results.best_comp.llm_score}/3</span><span class="vs-vector" data-tip="${tipVector}">🎯 Semantic Match: ${semMatch}%</span></div></div>
                    <div class="rag-ai-rationale">🤖 <strong>AI Rationale:</strong> ${results.best_comp.llm_reason}</div>
                    <div class="synthesis-box"><h5>AI Readiness & Data Report</h5>
                        <div class="synth-metric"><span class="synth-label">Density:</span> <span class="synth-value ${r.density.toLowerCase()}">${r.density}</span></div>
                        <div class="synth-metric"><span class="synth-label">Directness:</span> <span class="synth-value ${r.directness.toLowerCase()}">${r.directness}</span></div>
                        <div class="synth-metric"><span class="synth-label">Completeness:</span> <span class="synth-value ${r.completeness.toLowerCase()}">${r.completeness}</span></div>
                    </div>
                    <span class="chunk-label">Extracted Text (Parent):</span><div class="vs-quote">${results.best_comp.text}</div>
                 </div>`;
    }
    html += `</div>`; 

    if (results.optimized_rewrite) {
        html += `<div class="aio-rewrite-box">
                    <div class="aio-header-wrap"><h3>✨ AIO Content Re-writer (First-Party)</h3>
                    <button type="button" id="aio-copy-btn" class="copy-btn" onclick="copyAioRewrite()">Copy Content</button></div>
                    <p class="aio-rewrite-instructions">Copy this into your website (filling in the brackets) to secure the citation.</p>
                    <div id="aio-generated-text" class="aio-rewrite-content">${results.optimized_rewrite.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</div>
                 </div>`;
    }

    if (results.top_5_chunks.length) {
        html += `<div class="context-window-box"><h3>📚 The Context Window (Top 5 Ingredients)</h3>`;
        
        if (results.my_top_5_count === 0) {
            html += `<div class="rag-info-alert" style="margin-top: 15px; margin-bottom: 20px; border-left-color: var(--lose-red); background: #fef2f2; color: #991b1b;">
                        💡 <strong>Missing from the Top 5?</strong> Use the tool's <strong>AIO Content Re-writer</strong> above to replace your fluffy text with the highly dense, optimized rewrite it suggests!
                    </div>`;
        }
        
        html += `<div class="chunk-list">`;
        results.top_5_chunks.forEach((chunk, i) => {
            const domain = chunk.source === 'you' ? 'Your Site' : new URL(chunk.url).hostname;
            const youClass = chunk.source === 'you' ? 'source-you' : '';
            html += `<div class="chunk-item ${youClass}">
                        <div class="chunk-meta"><span class="chunk-rank">#${i+1}</span> <span class="chunk-domain">${domain}</span> <span class="chunk-score-badge">LLM Score: ${chunk.llm_score}/3</span></div>
                        <div class="chunk-text-preview">${chunk.text}</div>
                     </div>`;
        });
        html += `</div></div>`;
    }

    if (results.ai_overview) {
        html += `<div class="ai-overview-box">
                    <h3>✨ Simulated AI Overview</h3>
                    <div class="ai-overview-disclaimer"><strong>Disclaimer:</strong> This is the output of 1 of the 5 synthesis simulations we ran using your CURRENT website text.</div>
                    <div class="ai-overview-content">${results.ai_overview}</div>
                 </div>`;
    }

    html += `</div>`;
    return html;
}

app.listen(process.env.PORT || 3000, () => {
    console.log(`🚀 Advanced RAG Server is running on port ${process.env.PORT || 3000}`);
});
