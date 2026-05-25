/**
 * Brute Force Search - O(N) exact nearest neighbor search
 * Compares query vector against every stored vector
 */

class BruteForce {
    constructor() {
        this.items = [];
    }

    /**
     * Insert a vector item
     * @param {Object} item - { id, metadata, category, embedding }
     */
    insert(item) {
        this.items.push(item);
    }

    /**
     * Find k nearest neighbors using given distance function
     * @param {Array<number>} query - Query vector
     * @param {number} k - Number of results
     * @param {Function} distFn - Distance function
     * @returns {Array<{distance: number, id: number}>}
     */
    knn(query, k, distFn) {
        const results = [];

        for (const item of this.items) {
            const distance = distFn(query, item.embedding);
            results.push({ distance, id: item.id });
        }

        results.sort((a, b) => a.distance - b.distance);
        return results.slice(0, k);
    }

    /**
     * Remove item by ID
     * @param {number} id 
     */
    remove(id) {
        const index = this.items.findIndex(item => item.id === id);
        if (index !== -1) {
            this.items.splice(index, 1);
            return true;
        }
        return false;
    }

    /**
     * Get all items
     */
    getAll() {
        return this.items;
    }

    /**
     * Get size
     */
    size() {
        return this.items.length;
    }

    /**
     * Rebuild (clear and re-insert)
     * @param {Array} items 
     */
    rebuild(items) {
        this.items = [...items];
    }
}

module.exports = BruteForce;