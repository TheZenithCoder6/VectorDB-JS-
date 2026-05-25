# 🔍 VectorDB-JS

**A complete Vector Database built from scratch in JavaScript** — Understand how HNSW, KD-Tree, RAG, and document embeddings work by exploring a working system.

---

## 📌 Small Summary

VectorDB-JS is a fully functional vector database written entirely in Node.js/JavaScript. It converts text into numerical vectors and finds semantically similar content using three search algorithms (Brute Force, KD-Tree, HNSW). It also includes a complete RAG (Retrieval-Augmented Generation) pipeline powered by local Ollama models, allowing you to ask questions about your documents and get AI-powered answers.

---

## 🎯 What This Project Does

| Feature | Description |
|---------|-------------|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — compare speed vs accuracy |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization — watch semantic clusters form in real-time |
| **Real Document Embedding** | Paste any text → Ollama embeds it with nomic-embed-text (768D) |
| **RAG Pipeline** | Ask questions → HNSW retrieves relevant chunks → local LLM answers |
| **Full REST API** | CRUD operations, search, benchmark, hnsw-info endpoints |
| **Web Interface** | Interactive UI with tabs for search, documents, and AI chat |

---

## ⚙️ How It Works
┌─────────────────────────────────────────────────────────────────┐
│ YOUR TEXT │
│ "What is dynamic programming?" │
└─────────────────────────────┬───────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────────┐
│ OLLAMA (nomic-embed-text) │
│ Converts text to a 768-dimensional vector │
│ [0.12, -0.34, 0.56, -0.78, 0.90, ...] │
└─────────────────────────────┬───────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────────┐
│ HNSW INDEX │
│ Multi-layer graph — finds nearest neighbors in │
│ O(log N) time instead of O(N) │
└─────────────────────────────┬───────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────────┐
│ RETRIEVED CONTEXT │
│ Top 3 most semantically similar document chunks │
└─────────────────────────────┬───────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────────┐
│ OLLAMA (llama3.2) │
│ Generates an answer using the context │
└─────────────────────────────┬───────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────────┐
│ ANSWER │
│ "Dynamic programming is a method for solving complex │
│ problems by breaking them into simpler subproblems..." │
└─────────────────────────────────────────────────────────────────┘



**Simple Explanation:**
1. **Text → Numbers** — Ollama converts your words into a list of 768 numbers (vector)
2. **Find Similar** — HNSW finds the closest vectors in the database (semantic search)
3. **Retrieve Context** — Get the original text of those closest matches
4. **Generate Answer** — Send the context + question to llama3.2 for an AI answer

---

## 🛠️ What Has Been Used (with Explanations)

| Technology | Purpose | Why This? |
|------------|---------|------------|
| **Node.js** | Backend runtime | JavaScript everywhere — no language switching |
| **Express.js** | REST API server | Lightweight, fast HTTP server |
| **Axios** | HTTP client | Calls Ollama API from Node.js |
| **CORS** | Cross-origin middleware | Allows frontend to call backend |
| **Ollama** | Local AI model runner | Free, private, runs on your laptop |
| **nomic-embed-text** | Embedding model | Converts text → 768D vectors (274MB) |
| **llama3.2** | Language model | Generates answers (2GB, runs locally) |
| **Pure JavaScript** | All algorithms | No ML libraries — everything hand-coded |
| **Canvas API** | Scatter plot | Visualizes 16D vectors in 2D space |

**Algorithms Written From Scratch:**
- **HNSW** — Hierarchical Navigable Small World (multi-layer graph)
- **KD-Tree** — K-Dimensional Tree (space partitioning)
- **Brute Force** — Linear scan (baseline comparison)

---

## 📋 Prerequisites

| Requirement | Minimum Spec | Download Link |
|-------------|--------------|---------------|
| **Node.js** | v18 or later | [nodejs.org](https://nodejs.org) |
| **Ollama** | Any version | [ollama.com](https://ollama.com) |
| **RAM** | 8GB (16GB recommended) | — |
| **Disk Space** | 3GB free | For AI models |

**Required Ollama Models:**
```bash
ollama pull nomic-embed-text   # 274MB — for embeddings
ollama pull llama3.2            # 2GB — for text generation


💻 Step-by-Step Setup (Windows)
step 1 — Install Node.js
Go to https://nodejs.org

Download LTS version (18.x or later)

Run the installer with default settings

Verify installation in PowerShell:

powershell
node --version   # Should show v18.x.x or higher
npm --version    # Should show 9.x.x or higher

ollama pull nomic-embed-text
ollama pull llama3.2

Step 3 — Download the Project
Option A — Download ZIP:

Click the green "Code" button above

Select "Download ZIP"

Extract to C:\VectorDB-JS

npm install
Step 5 — Start Ollama (if not running)
ollama serve


Step 6 — Start VectorDB Server
Open another PowerShell in the project folder:

powershell
npm start
You should see:

text
=== VectorDB Engine (JavaScript) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
Server running on http://localhost:8080
Step 7 — Open Web Interface
Open your browser and go to:

text
http://localhost:8080
🔌 REST API Reference
Demo Vector Endpoints
Method	Endpoint	Description	Example
GET	/search	K-NN search	/search?v=0.9,0.8,...&k=5&metric=cosine&algo=hnsw
POST	/insert	Insert vector	{"metadata":"x","category":"cs","embedding":[0.1,0.2,...]}
DELETE	/delete/:id	Delete by ID	/delete/5
GET	/items	List all vectors	/items
GET	/benchmark	Compare algorithms	/benchmark?v=0.9,0.8,...&k=5
GET	/hnsw-info	HNSW graph stats	/hnsw-info
GET	/stats	DB statistics	/stats
Document & RAG Endpoints
Method	Endpoint	Body	Description
POST	/doc/insert	{"title":"...","text":"..."}	Embed & store document
GET	/doc/list	—	List all documents
DELETE	/doc/delete/:id	—	Delete document chunk
POST	/doc/search	{"question":"...","k":3}	Search (no LLM)
POST	/doc/ask	{"question":"...","k":3}	Full RAG pipeline
GET	/status	—	Ollama status & model info
📂 Project Structure
text
VectorDB-JS/
│
├── server.js              # Main Express server (all API endpoints)
├── hnsw.js                # HNSW algorithm (multi-layer graph search)
├── kdtree.js              # KD-Tree algorithm (space partitioning)
├── bruteforce.js          # Brute force search (baseline)
├── ollama.js              # Ollama API client (embeddings + generation)
├── package.json           # Node.js dependencies
│
├── public/                # Static frontend files
│   └── index.html         # Web UI (scatter plot, chat, document manager)
│
└── README.md              # This file
🏗️ Architecture
text
┌────────────────────────────────────────────────────────────────────┐
│                         WEB BROWSER                                │
│                    (index.html + Canvas + JS)                      │
│                                                                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│   │  Search  │  │ Documents│  │  Ask AI  │  │ Scatter  │          │
│   │   Tab    │  │   Tab    │  │   Tab    │  │   Plot   │          │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│        └─────────────┴─────────────┴─────────────┘                  │
│                         │ HTTP/REST API                             │
└─────────────────────────┼──────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────────┐
│                      EXPRESS SERVER (server.js)                     │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    ROUTE HANDLERS                            │  │
│   │  /search  /insert  /delete  /benchmark  /hnsw-info  /status │  │
│   │  /doc/insert  /doc/list  /doc/delete  /doc/search  /doc/ask │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│   │  DemoVectorDB│  │  DocumentDB  │  │  OllamaClient│             │
│   │   (16D)      │  │   (768D)     │  │  (HTTP)      │             │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│          │                 │                 │                     │
│          ▼                 ▼                 ▼                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    ALGORITHMS (from scratch)                 │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │  │
│   │  │  HNSW    │  │  KD-Tree │  │  Brute   │                   │  │
│   │  │ (graph)  │  │ (space)  │  │  Force   │                   │  │
│   │  └──────────┘  └──────────┘  └──────────┘                   │  │
│   └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                          │
                          │ HTTP
                          ▼
┌────────────────────────────────────────────────────────────────────┐
│                         OLLAMA SERVER                               │
│                     (http://localhost:11434)                        │
│                                                                     │
│   ┌─────────────────────┐    ┌─────────────────────┐               │
│   │  nomic-embed-text   │    │     llama3.2        │               │
│   │  (embeddings 768D)  │    │  (text generation)  │               │
│   └─────────────────────┘    └─────────────────────┘               │
└────────────────────────────────────────────────────────────────────┘
🧠 Algorithm Deep Dive
1. HNSW (Hierarchical Navigable Small World)
What it is: A multi-layer graph structure where each layer is progressively sparser. Search starts at the top layer (few nodes, long jumps) and zooms in.

How it works:

text
Layer 2:    ●─────●         (4 nodes, long connections)
             \   /
Layer 1:    ●─●─●─●───●      (more nodes, medium connections)
             |/ \| 
Layer 0:    ●─●─●─●─●─●─●    (all nodes, short connections)
Insert: Each node gets a random level (exponentially fewer nodes at higher layers). For each level from top to bottom, find nearest neighbors and connect bidirectionally.

Search: Greedy descent from top layer → at layer 0, expand search using a priority queue (beam search).

Time Complexity: O(log N) · O(M) where M is number of connections per node.

Why it's fast: Upper layers act like express highways — you quickly jump to the right neighborhood.

2. KD-Tree (K-Dimensional Tree)
What it is: A binary space-partitioning tree. Each node splits the space along one dimension, creating axis-aligned bounding boxes.

How it works:

text
           Root (dim 0, value 0.5)
          /                      \
    Left (dim 1, 0.3)        Right (dim 1, 0.7)
       /        \              /        \
    Leaf      Leaf          Leaf       Leaf
Search: Traverse down to the leaf containing the query point. Then check if any other subtree could contain a closer point using the "ball within hyperslab" heuristic — prune entire branches when the closest possible distance is worse than current best.

Time Complexity: O(log N) for low dimensions (≤20), degrades to O(N) at 100+ dimensions.

Weakness: The "curse of dimensionality" — in high dimensions, almost all space is near the boundary, so no branches get pruned.

3. Brute Force
What it is: Linear scan through all vectors. Compute distance to every single vector and keep the k smallest.

How it works:

text
for each vector in database:
    dist = distance(query, vector)
    if dist < best_k_distances[k-1]:
        insert into results
Time Complexity: O(N · d) where N = number of vectors, d = dimensions.

When to use: Small datasets (<1000 vectors), exact results required, or as a baseline for benchmarking.

4. Distance Metrics
Metric	Formula	Best For	Range
Cosine	1 - (A·B)/(|A||B|)	Text similarity, document search	0 (identical) to 2 (opposite)
Euclidean	√Σ(Ai-Bi)²	Spatial data, image vectors	0 to ∞
Manhattan	Σ|Ai-Bi|	Grid-based data, robust to outliers	0 to ∞
When to use Cosine: Always for text embeddings (like this project) because it focuses on direction (meaning) not magnitude (length).

5. RAG Pipeline (Retrieval-Augmented Generation)
text
┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PHASE                               │
│                                                                  │
│  Question ──► Embed ──► HNSW Search ──► Top-k Chunks            │
│  "What is   │          │              │                          │
│   HNSW?"    ▼          ▼              ▼                          │
│          [768D]    O(log N)      [chunk1, chunk2, chunk3]       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATION PHASE                              │
│                                                                  │
│  Context + Question ──► Prompt ──► llama3.2 ──► Answer          │
│                                                                  │
│  Prompt: "Use the context to answer: [chunks]. Question: ..."   │
└─────────────────────────────────────────────────────────────────┘
Why RAG matters: Instead of retraining the LLM on your documents, you just retrieve relevant chunks at query time. This is cheaper, faster, and works with any document collection.


📄 License
MIT License — not for comersial use, modify, and distribute for any purpose.

<div align="center">
Built with ❤️ to demystify Vector Databases

[⭐ Star this repo] | [🐛 Report Issue] | [📖 Read Docs]

</div> ```
