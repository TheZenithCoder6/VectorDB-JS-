/**
 * KD-Tree (K-Dimensional Tree) - Space-partitioning data structure
 * For 16D vectors, efficient up to ~20 dimensions
 */

class KDNode {
    constructor(item) {
        this.item = item;
        this.left = null;
        this.right = null;
    }
}

class KDTree {
    constructor(dims) {
        this.dims = dims;
        this.root = null;
    }

    /**
     * Insert a vector into the tree
     * @param {Object} item 
     */
    insert(item) {
        this.root = this._insertRec(this.root, item, 0);
    }

    _insertRec(node, item, depth) {
        if (node === null) {
            return new KDNode(item);
        }

        const axis = depth % this.dims;
        const nodeVal = node.item.embedding[axis];
        const itemVal = item.embedding[axis];

        if (itemVal < nodeVal) {
            node.left = this._insertRec(node.left, item, depth + 1);
        } else {
            node.right = this._insertRec(node.right, item, depth + 1);
        }

        return node;
    }

    /**
     * Find k nearest neighbors
     * @param {Array<number>} query 
     * @param {number} k 
     * @param {Function} distFn 
     * @returns {Array<{distance: number, id: number}>}
     */
    knn(query, k, distFn) {
        const heap = []; // Max-heap simulation (store as array, keep size <= k)
        
        this._knnRec(this.root, query, k, 0, distFn, heap);
        
        // Convert heap to sorted results
        const results = heap.map(item => ({ distance: item.distance, id: item.id }));
        results.sort((a, b) => a.distance - b.distance);
        return results;
    }

    _knnRec(node, query, k, depth, distFn, heap) {
        if (node === null) return;

        const distance = distFn(query, node.item.embedding);
        
        // Add to heap (keep only best k)
        heap.push({ distance, id: node.item.id });
        heap.sort((a, b) => b.distance - a.distance); // Max-heap order
        if (heap.length > k) heap.pop();

        const axis = depth % this.dims;
        const diff = query[axis] - node.item.embedding[axis];
        
        // Determine which side to search first
        const closer = diff < 0 ? node.left : node.right;
        const farther = diff < 0 ? node.right : node.left;
        
        this._knnRec(closer, query, k, depth + 1, distFn, heap);
        
        // Check if we need to search the other side
        if (heap.length < k || Math.abs(diff) < heap[0].distance) {
            this._knnRec(farther, query, k, depth + 1, distFn, heap);
        }
    }

    /**
     * Rebuild tree from scratch
     * @param {Array} items 
     */
    rebuild(items) {
        this.root = null;
        for (const item of items) {
            this.insert(item);
        }
    }

    /**
     * Delete all nodes (for garbage collection)
     */
    clear() {
        this.root = null;
    }
}

module.exports = KDTree;