/**
 * HNSW (Hierarchical Navigable Small World)
 * Production-grade approximate nearest neighbor search
 * O(log N) complexity, works well in high dimensions
 */

class HNSWNode {
    constructor(item, maxLevel) {
        this.id = item.id;
        this.item = item;
        this.maxLevel = maxLevel;
        // neighbors[level] = array of neighbor IDs
        this.neighbors = Array(maxLevel + 1).fill().map(() => []);
    }
}

class HNSW {
    constructor(options = {}) {
        this.M = options.M || 16;           // Max connections per layer
        this.M0 = options.M0 || 32;         // Max connections at layer 0
        this.efConstruction = options.efConstruction || 200;
        this.ml = options.ml || (1.0 / Math.log(this.M));
        this.rngSeed = options.rngSeed || 42;
        
        this.graph = new Map();              // id -> HNSWNode
        this.entryPoint = null;
        this.topLayer = -1;
        this.rng = this._createRng(this.rngSeed);
    }

    _createRng(seed) {
        // Simple deterministic RNG (mulberry32)
        return function() {
            let t = seed += 0x6D2B79F5;
            t = Math.imul(t ^ t >>> 15, t | 1);
            t ^= t + Math.imul(t ^ t >>> 7, t | 61);
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
        };
    }

    _randomLevel() {
        const r = this.rng();
        return Math.floor(-Math.log(r) * this.ml);
    }

    /**
     * Search a single layer for nearest neighbors
     * @returns {Array<{distance: number, id: number}>}
     */
    _searchLayer(query, epId, ef, level, distFn) {
        const visited = new Set();
        const candidates = []; // Min-heap by distance
        const results = [];    // Max-heap by distance (store best)

        const ep = this.graph.get(epId);
        const epDist = distFn(query, ep.item.embedding);
        
        visited.add(epId);
        this._heapPush(candidates, { distance: epDist, id: epId }, true);
        this._heapPush(results, { distance: epDist, id: epId }, false);

        while (candidates.length > 0) {
            const current = this._heapPop(candidates, true);
            
            if (results.length >= ef && current.distance > results[0].distance) {
                break;
            }

            const node = this.graph.get(current.id);
            if (!node || level >= node.neighbors.length) continue;

            for (const neighborId of node.neighbors[level]) {
                if (visited.has(neighborId)) continue;
                visited.add(neighborId);

                const neighbor = this.graph.get(neighborId);
                if (!neighbor) continue;

                const dist = distFn(query, neighbor.item.embedding);
                
                if (results.length < ef || dist < results[0].distance) {
                    this._heapPush(candidates, { distance: dist, id: neighborId }, true);
                    this._heapPush(results, { distance: dist, id: neighborId }, false);
                    
                    if (results.length > ef) {
                        this._heapPop(results, false);
                    }
                }
            }
        }

        // Convert to sorted array
        const sorted = [];
        while (results.length) {
            sorted.push(this._heapPop(results, false));
        }
        sorted.sort((a, b) => a.distance - b.distance);
        return sorted;
    }

    _heapPush(heap, item, isMinHeap) {
        heap.push(item);
        let idx = heap.length - 1;
        
        while (idx > 0) {
            const parentIdx = Math.floor((idx - 1) / 2);
            const parent = heap[parentIdx];
            const better = isMinHeap 
                ? item.distance < parent.distance
                : item.distance > parent.distance;
            
            if (better) {
                [heap[idx], heap[parentIdx]] = [heap[parentIdx], heap[idx]];
                idx = parentIdx;
            } else {
                break;
            }
        }
    }

    _heapPop(heap, isMinHeap) {
        if (heap.length === 1) return heap.pop();
        
        const top = heap[0];
        heap[0] = heap.pop();
        
        let idx = 0;
        while (true) {
            let bestIdx = idx;
            const leftIdx = 2 * idx + 1;
            const rightIdx = 2 * idx + 2;
            
            if (leftIdx < heap.length) {
                const better = isMinHeap
                    ? heap[leftIdx].distance < heap[bestIdx].distance
                    : heap[leftIdx].distance > heap[bestIdx].distance;
                if (better) bestIdx = leftIdx;
            }
            
            if (rightIdx < heap.length) {
                const better = isMinHeap
                    ? heap[rightIdx].distance < heap[bestIdx].distance
                    : heap[rightIdx].distance > heap[bestIdx].distance;
                if (better) bestIdx = rightIdx;
            }
            
            if (bestIdx === idx) break;
            
            [heap[idx], heap[bestIdx]] = [heap[bestIdx], heap[idx]];
            idx = bestIdx;
        }
        
        return top;
    }

    _selectNeighbors(candidates, maxM) {
        // Simple selection: take closest maxM candidates
        candidates.sort((a, b) => a.distance - b.distance);
        return candidates.slice(0, maxM).map(c => c.id);
    }

    /**
     * Insert a vector into HNSW
     * @param {Object} item - { id, metadata, category, embedding }
     * @param {Function} distFn 
     */
    insert(item, distFn) {
        const id = item.id;
        const level = this._randomLevel();
        const newNode = new HNSWNode(item, level);
        this.graph.set(id, newNode);

        // First node - initialize as entry point
        if (this.entryPoint === null) {
            this.entryPoint = id;
            this.topLayer = level;
            return;
        }

        let ep = this.entryPoint;
        
        // Greedy descent from top layer to level+1
        for (let lc = this.topLayer; lc > level; lc--) {
            const node = this.graph.get(ep);
            if (node && lc < node.neighbors.length) {
                const W = this._searchLayer(item.embedding, ep, 1, lc, distFn);
                if (W.length > 0) ep = W[0].id;
            }
        }

        // Insert at each layer from min(topLayer, level) down to 0
        for (let lc = Math.min(this.topLayer, level); lc >= 0; lc--) {
            const W = this._searchLayer(item.embedding, ep, this.efConstruction, lc, distFn);
            const maxM = (lc === 0) ? this.M0 : this.M;
            const neighbors = this._selectNeighbors(W, maxM);
            
            newNode.neighbors[lc] = neighbors;

            // Bidirectional connections
            for (const neighborId of neighbors) {
                const neighbor = this.graph.get(neighborId);
                if (!neighbor) continue;
                
                if (neighbor.neighbors.length <= lc) {
                    neighbor.neighbors = neighbor.neighbors.concat(
                        Array(lc + 1 - neighbor.neighbors.length).fill().map(() => [])
                    );
                }
                
                neighbor.neighbors[lc].push(id);
                
                // Shrink if too many connections
                if (neighbor.neighbors[lc].length > maxM) {
                    // Recompute distances to neighbors and keep closest maxM
                    const distances = [];
                    for (const nid of neighbor.neighbors[lc]) {
                        const nnode = this.graph.get(nid);
                        if (nnode) {
                            const d = distFn(neighbor.item.embedding, nnode.item.embedding);
                            distances.push({ distance: d, id: nid });
                        }
                    }
                    distances.sort((a, b) => a.distance - b.distance);
                    neighbor.neighbors[lc] = distances.slice(0, maxM).map(d => d.id);
                }
            }
            
            if (W.length > 0) ep = W[0].id;
        }

        if (level > this.topLayer) {
            this.topLayer = level;
            this.entryPoint = id;
        }
    }

    /**
     * Find k nearest neighbors
     * @param {Array<number>} query 
     * @param {number} k 
     * @param {number} ef - Search ef (higher = more accurate but slower)
     * @param {Function} distFn 
     * @returns {Array<{distance: number, id: number}>}
     */
    knn(query, k, ef, distFn) {
        if (this.entryPoint === null) return [];

        let ep = this.entryPoint;
        
        // Greedy descent through upper layers
        for (let lc = this.topLayer; lc > 0; lc--) {
            const node = this.graph.get(ep);
            if (node && lc < node.neighbors.length) {
                const W = this._searchLayer(query, ep, 1, lc, distFn);
                if (W.length > 0) ep = W[0].id;
            }
        }
        
        // Search layer 0 with ef parameter
        const results = this._searchLayer(query, ep, Math.max(ef, k), 0, distFn);
        return results.slice(0, k);
    }

    /**
     * Remove a node by ID
     * @param {number} id 
     */
    remove(id) {
        if (!this.graph.has(id)) return false;
        
        // Remove from all neighbor lists
        for (const [nid, node] of this.graph) {
            for (const layer of node.neighbors) {
                const idx = layer.indexOf(id);
                if (idx !== -1) layer.splice(idx, 1);
            }
        }
        
        // Update entry point if needed
        if (this.entryPoint === id) {
            this.entryPoint = null;
            for (const [nid] of this.graph) {
                if (nid !== id) {
                    this.entryPoint = nid;
                    break;
                }
            }
        }
        
        this.graph.delete(id);
        return true;
    }

    /**
     * Get graph information for visualization
     */
    getInfo() {
        const nodesPerLayer = [];
        const edgesPerLayer = [];
        
        for (const [id, node] of this.graph) {
            for (let lc = 0; lc <= node.maxLevel; lc++) {
                while (nodesPerLayer.length <= lc) {
                    nodesPerLayer.push(0);
                    edgesPerLayer.push(0);
                }
                nodesPerLayer[lc]++;
                if (lc < node.neighbors.length) {
                    edgesPerLayer[lc] += node.neighbors[lc].length;
                }
            }
        }
        
        // Halve edges (each edge counted twice)
        for (let lc = 0; lc < edgesPerLayer.length; lc++) {
            edgesPerLayer[lc] = Math.floor(edgesPerLayer[lc] / 2);
        }
        
        return {
            topLayer: this.topLayer,
            nodeCount: this.graph.size,
            nodesPerLayer,
            edgesPerLayer
        };
    }

    size() {
        return this.graph.size;
    }
}

module.exports = HNSW;