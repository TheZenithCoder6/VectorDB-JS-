/**
 * VectorDB - Complete JavaScript Implementation
 * Express server with HNSW, KD-Tree, Brute Force, and RAG pipeline
 */

const express = require('express');
const cors = require('cors');
const path = require('path');

const BruteForce = require('./bruteforce');
const KDTree = require('./kdtree');
const HNSW = require('./hnsw');
const OllamaClient = require('./ollama');

const app = express();
const PORT = 8080;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static('public'));

// =====================================================================
//  CONSTANTS
// =====================================================================

const DEMO_DIMS = 16;
const DOC_DIMS = 768;  // nomic-embed-text output dimension

// =====================================================================
//  DISTANCE FUNCTIONS
// =====================================================================

function euclidean(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        sum += d * d;
    }
    return Math.sqrt(sum);
}

function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-9 || nb < 1e-9) return 1.0;
    return 1.0 - dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function manhattan(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += Math.abs(a[i] - b[i]);
    }
    return sum;
}

function getDistFn(metric) {
    switch (metric) {
        case 'cosine': return cosine;
        case 'manhattan': return manhattan;
        default: return euclidean;
    }
}

// =====================================================================
//  DEMO VECTOR DATABASE (16D)
// =====================================================================

class DemoVectorDB {
    constructor() {
        this.store = new Map();      // id -> VectorItem
        this.bruteForce = new BruteForce();
        this.kdTree = new KDTree(DEMO_DIMS);
        this.hnsw = new HNSW({ M: 16, efConstruction: 200 });
        this.nextId = 1;
    }

    insert(metadata, category, embedding) {
        const id = this.nextId++;
        const item = { id, metadata, category, embedding };
        
        this.store.set(id, item);
        this.bruteForce.insert(item);
        this.kdTree.insert(item);
        this.hnsw.insert(item, cosine);
        
        return id;
    }

    remove(id) {
        if (!this.store.has(id)) return false;
        
        this.store.delete(id);
        this.bruteForce.remove(id);
        this.hnsw.remove(id);
        
        // Rebuild KD-Tree (simpler than implementing delete in KD-Tree)
        const remaining = Array.from(this.store.values());
        this.kdTree.clear();
        this.kdTree.rebuild(remaining);
        
        return true;
    }

    search(query, k, metric, algorithm) {
        const startTime = process.hrtime.bigint();
        const distFn = getDistFn(metric);
        
        let rawResults;
        switch (algorithm) {
            case 'bruteforce':
                rawResults = this.bruteForce.knn(query, k, distFn);
                break;
            case 'kdtree':
                rawResults = this.kdTree.knn(query, k, distFn);
                break;
            default:  // hnsw
                rawResults = this.hnsw.knn(query, k, 50, distFn);
                break;
        }
        
        const endTime = process.hrtime.bigint();
        const latencyUs = Number(endTime - startTime) / 1000;
        
        const hits = [];
        for (const r of rawResults) {
            const item = this.store.get(r.id);
            if (item) {
                hits.push({
                    id: item.id,
                    metadata: item.metadata,
                    category: item.category,
                    distance: r.distance,
                    embedding: item.embedding
                });
            }
        }
        
        return { hits, latencyUs, algorithm, metric };
    }

    benchmark(query, k, metric) {
        const distFn = getDistFn(metric);
        
        const timeFn = (fn) => {
            const start = process.hrtime.bigint();
            fn();
            const end = process.hrtime.bigint();
            return Number(end - start) / 1000;
        };
        
        return {
            bruteforceUs: timeFn(() => this.bruteForce.knn(query, k, distFn)),
            kdtreeUs: timeFn(() => this.kdTree.knn(query, k, distFn)),
            hnswUs: timeFn(() => this.hnsw.knn(query, k, 50, distFn)),
            itemCount: this.store.size
        };
    }

    getAll() {
        return Array.from(this.store.values());
    }

    size() {
        return this.store.size;
    }

    hnswInfo() {
        return this.hnsw.getInfo();
    }
}

// =====================================================================
//  DOCUMENT DATABASE (Real Ollama Embeddings)
// =====================================================================

class DocumentDB {
    constructor() {
        this.store = new Map();   // id -> DocItem
        this.hnsw = new HNSW({ M: 16, efConstruction: 200 });
        this.bruteForce = new BruteForce();
        this.nextId = 1;
        this.dims = 0;
    }

    insert(title, text, embedding) {
        if (this.dims === 0) this.dims = embedding.length;
        
        const id = this.nextId++;
        const item = { id, title, text, embedding };
        
        this.store.set(id, item);
        
        const vectorItem = { id, metadata: title, category: 'doc', embedding };
        this.hnsw.insert(vectorItem, cosine);
        this.bruteForce.insert(vectorItem);
        
        return id;
    }

    async search(queryEmbedding, k, maxDistance = 0.7) {
        if (this.store.size === 0) return [];
        
        const rawResults = this.store.size < 10
            ? this.bruteForce.knn(queryEmbedding, k, cosine)
            : this.hnsw.knn(queryEmbedding, k, 50, cosine);
        
        const results = [];
        for (const r of rawResults) {
            const item = this.store.get(r.id);
            if (item && r.distance <= maxDistance) {
                results.push({ distance: r.distance, item });
            }
        }
        return results;
    }

    remove(id) {
        if (!this.store.has(id)) return false;
        
        this.store.delete(id);
        this.hnsw.remove(id);
        this.bruteForce.remove(id);
        return true;
    }

    getAll() {
        return Array.from(this.store.values());
    }

    size() {
        return this.store.size;
    }

    getDims() {
        return this.dims;
    }
}

// =====================================================================
//  TEXT CHUNKER
// =====================================================================

function chunkText(text, chunkWords = 250, overlapWords = 30) {
    const words = text.split(/\s+/);
    if (words.length === 0) return [];
    if (words.length <= chunkWords) return [text];
    
    const chunks = [];
    const step = chunkWords - overlapWords;
    
    for (let i = 0; i < words.length; i += step) {
        const end = Math.min(i + chunkWords, words.length);
        const chunk = words.slice(i, end).join(' ');
        chunks.push(chunk);
        if (end === words.length) break;
    }
    
    return chunks;
}

// =====================================================================
//  INITIALIZE DATABASES
// =====================================================================

const demoDB = new DemoVectorDB();
const docDB = new DocumentDB();
const ollama = new OllamaClient();

// Load demo data
function loadDemoData() {
    const demoItems = [
        ["Linked List: nodes connected by pointers", "cs",
            [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]],
        ["Binary Search Tree: O(log n) search and insert", "cs",
            [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]],
        ["Dynamic Programming: memoization overlapping subproblems", "cs",
            [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]],
        ["Graph BFS and DFS: breadth and depth first traversal", "cs",
            [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]],
        ["Hash Table: O(1) lookup with collision chaining", "cs",
            [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]],
        ["Calculus: derivatives integrals and limits", "math",
            [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]],
        ["Linear Algebra: matrices eigenvalues eigenvectors", "math",
            [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]],
        ["Probability: distributions random variables Bayes theorem", "math",
            [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]],
        ["Number Theory: primes modular arithmetic RSA cryptography", "math",
            [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]],
        ["Combinatorics: permutations combinations generating functions", "math",
            [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]],
        ["Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
            [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]],
        ["Sushi: vinegared rice raw fish and nori rolls", "food",
            [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]],
        ["Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
            [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]],
        ["Tacos: corn tortillas with carnitas salsa and cilantro", "food",
            [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]],
        ["Croissant: laminated pastry with buttery flaky layers", "food",
            [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]],
        ["Basketball: fast-paced shooting dribbling slam dunks", "sports",
            [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]],
        ["Football: tackles touchdowns field goals and strategy", "sports",
            [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]],
        ["Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
            [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]],
        ["Chess: openings endgames tactics strategic board game", "sports",
            [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]],
        ["Swimming: butterfly freestyle backstroke Olympic competition", "sports",
            [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]]
    ];
    
    for (const [meta, cat, emb] of demoItems) {
        demoDB.insert(meta, cat, emb);
    }
    console.log(`Loaded ${demoItems.length} demo vectors | ${DEMO_DIMS} dims`);
}

// =====================================================================
//  HELPER FUNCTIONS
// =====================================================================

function parseVector(str) {
    const parts = str.split(',');
    const vec = [];
    for (const p of parts) {
        const val = parseFloat(p);
        if (!isNaN(val)) vec.push(val);
    }
    return vec;
}

function escapeJson(str) {
    return str.replace(/\\/g, '\\\\')
              .replace(/"/g, '\\"')
              .replace(/\n/g, '\\n')
              .replace(/\r/g, '\\r')
              .replace(/\t/g, '\\t');
}

// =====================================================================
//  API ENDPOINTS
// =====================================================================

// Demo Vector Endpoints

app.get('/search', (req, res) => {
    const vecStr = req.query.v;
    if (!vecStr) {
        return res.status(400).json({ error: 'Missing vector parameter' });
    }
    
    const query = parseVector(vecStr);
    if (query.length !== DEMO_DIMS) {
        return res.status(400).json({ 
            error: `Need ${DEMO_DIMS}D vector, got ${query.length}D` 
        });
    }
    
    let k = parseInt(req.query.k) || 5;
    k = Math.min(k, 20);
    const metric = req.query.metric || 'cosine';
    const algorithm = req.query.algo || 'hnsw';
    
    const result = demoDB.search(query, k, metric, algorithm);
    res.json(result);
});

app.post('/insert', (req, res) => {
    const { metadata, category, embedding } = req.body;
    
    if (!metadata || !category || !embedding || embedding.length !== DEMO_DIMS) {
        return res.status(400).json({ error: 'Invalid body' });
    }
    
    const id = demoDB.insert(metadata, category, embedding);
    res.json({ id });
});

app.delete('/delete/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const ok = demoDB.remove(id);
    res.json({ ok });
});

app.get('/items', (req, res) => {
    const items = demoDB.getAll();
    res.json(items);
});

app.get('/benchmark', (req, res) => {
    const vecStr = req.query.v;
    if (!vecStr) {
        return res.status(400).json({ error: 'Missing vector parameter' });
    }
    
    const query = parseVector(vecStr);
    if (query.length !== DEMO_DIMS) {
        return res.status(400).json({ 
            error: `Need ${DEMO_DIMS}D vector` 
        });
    }
    
    let k = parseInt(req.query.k) || 5;
    k = Math.min(k, 20);
    const metric = req.query.metric || 'cosine';
    
    const result = demoDB.benchmark(query, k, metric);
    res.json(result);
});

app.get('/hnsw-info', (req, res) => {
    const info = demoDB.hnswInfo();
    res.json(info);
});

app.get('/stats', (req, res) => {
    res.json({
        count: demoDB.size(),
        dims: DEMO_DIMS,
        algorithms: ['bruteforce', 'kdtree', 'hnsw'],
        metrics: ['euclidean', 'cosine', 'manhattan']
    });
});

// Document & RAG Endpoints

app.post('/doc/insert', async (req, res) => {
    const { title, text } = req.body;
    
    if (!title || !text) {
        return res.status(400).json({ error: 'Need title and text' });
    }
    
    const chunks = chunkText(text, 250, 30);
    const ids = [];
    
    for (let i = 0; i < chunks.length; i++) {
        const embedding = await ollama.embed(chunks[i]);
        
        if (embedding.length === 0) {
            return res.status(503).json({ 
                error: 'Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text'
            });
        }
        
        const chunkTitle = chunks.length > 1
            ? `${title} [${i+1}/${chunks.length}]`
            : title;
        
        const id = docDB.insert(chunkTitle, chunks[i], embedding);
        ids.push(id);
    }
    
    res.json({
        ids,
        chunks: chunks.length,
        dims: docDB.getDims()
    });
});

app.delete('/doc/delete/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const ok = docDB.remove(id);
    res.json({ ok });
});

app.get('/doc/list', (req, res) => {
    const docs = docDB.getAll();
    const result = docs.map(doc => ({
        id: doc.id,
        title: doc.title,
        preview: doc.text.substring(0, 120) + (doc.text.length > 120 ? '…' : ''),
        words: doc.text.split(/\s+/).length
    }));
    res.json(result);
});

app.post('/doc/search', async (req, res) => {
    const { question, k = 3 } = req.body;
    
    if (!question) {
        return res.status(400).json({ error: 'Need question' });
    }
    
    const qEmb = await ollama.embed(question);
    if (qEmb.length === 0) {
        return res.status(503).json({ error: 'Ollama unavailable' });
    }
    
    const hits = await docDB.search(qEmb, k);
    const contexts = hits.map(hit => ({
        id: hit.item.id,
        title: hit.item.title,
        distance: hit.distance
    }));
    
    res.json({ contexts });
});

app.post('/doc/ask', async (req, res) => {
    const { question, k = 3 } = req.body;
    
    if (!question) {
        return res.status(400).json({ error: 'Need question' });
    }
    
    // Step 1: Embed the question
    const qEmb = await ollama.embed(question);
    if (qEmb.length === 0) {
        return res.status(503).json({ 
            error: 'Ollama unavailable. Please ensure Ollama is running and nomic-embed-text is installed.'
        });
    }
    
    // Step 2: Retrieve relevant chunks
    const hits = await docDB.search(qEmb, k);
    
    // Step 3: Build prompt
    let contextStr = '';
    for (let i = 0; i < hits.length; i++) {
        contextStr += `[${i+1}] ${hits[i].item.title}:\n${hits[i].item.text}\n\n`;
    }
    
    const prompt = `You are a helpful assistant. Answer the user's question directly. Use the provided context if it contains relevant information. If it doesn't, just use your own general knowledge. IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. Just answer the question naturally.

Context:
${contextStr}
Question: ${question}

Answer:`;
    
    // Step 4: Generate answer
    const answer = await ollama.generate(prompt);
    
    // Step 5: Return everything
    const contexts = hits.map(hit => ({
        id: hit.item.id,
        title: hit.item.title,
        text: hit.item.text,
        distance: hit.distance
    }));
    
    res.json({
        answer,
        model: ollama.genModel,
        contexts,
        docCount: docDB.size()
    });
});

app.get('/status', async (req, res) => {
    const available = await ollama.isAvailable();
    res.json({
        ollamaAvailable: available,
        embedModel: ollama.embedModel,
        genModel: ollama.genModel,
        docCount: docDB.size(),
        docDims: docDB.getDims(),
        demoDims: DEMO_DIMS,
        demoCount: demoDB.size()
    });
});

// Static files
app.use(express.static('public'));

// =====================================================================
//  START SERVER
// =====================================================================

async function main() {
    loadDemoData();
    
    const ollamaAvailable = await ollama.isAvailable();
    
    console.log('\n=== VectorDB Engine (JavaScript) ===');
    console.log(`http://localhost:${PORT}`);
    console.log(`${demoDB.size()} demo vectors | ${DEMO_DIMS} dims | HNSW+KD-Tree+BruteForce`);
    console.log(`Ollama: ${ollamaAvailable ? 'ONLINE' : 'OFFLINE (install from ollama.com)'}`);
    if (ollamaAvailable) {
        console.log(`  embed model: ${ollama.embedModel}  gen model: ${ollama.genModel}`);
    }
    console.log('');
    
    app.listen(PORT, '0.0.0.0', () => {
        console.log(`Server running on http://localhost:${PORT}`);
    });
}

main();