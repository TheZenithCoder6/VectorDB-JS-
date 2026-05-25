/**
 * Ollama API Client
 * Handles embedding generation and text generation via local Ollama
 */

const axios = require('axios');

class OllamaClient {
    constructor(host = '127.0.0.1', port = 11434) {
        this.baseUrl = `http://${host}:${port}`;
        this.embedModel = 'nomic-embed-text';
        this.genModel = 'llama3.2';
        this.timeout = 30000;
    }

    /**
     * Escape string for JSON embedding
     */
    _escapeJson(str) {
        return str.replace(/\\/g, '\\\\')
                  .replace(/"/g, '\\"')
                  .replace(/\n/g, '\\n')
                  .replace(/\r/g, '\\r')
                  .replace(/\t/g, '\\t');
    }

    /**
     * Check if Ollama is available
     */
    async isAvailable() {
        try {
            const response = await axios.get(`${this.baseUrl}/api/tags`, {
                timeout: 2000
            });
            return response.status === 200;
        } catch (error) {
            return false;
        }
    }

    /**
     * Get embedding for text using nomic-embed-text
     * @param {string} text 
     * @returns {Promise<Array<number>>}
     */
    async embed(text) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/embeddings`, {
                model: this.embedModel,
                prompt: text
            }, {
                timeout: this.timeout,
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.data && response.data.embedding) {
                return response.data.embedding;
            }
            return [];
        } catch (error) {
            console.error('Ollama embed error:', error.message);
            return [];
        }
    }

    /**
     * Generate text response using language model
     * @param {string} prompt 
     * @returns {Promise<string>}
     */
    async generate(prompt) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/generate`, {
                model: this.genModel,
                prompt: prompt,
                stream: false
            }, {
                timeout: 180000,  // 3 minutes for LLM generation
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.data && response.data.response) {
                return response.data.response;
            }
            return "ERROR: No response from Ollama";
        } catch (error) {
            console.error('Ollama generate error:', error.message);
            return "ERROR: Ollama unavailable. Please ensure Ollama is running and the model is downloaded.";
        }
    }

    /**
     * Set models (for runtime configuration)
     */
    setModels(embedModel, genModel) {
        if (embedModel) this.embedModel = embedModel;
        if (genModel) this.genModel = genModel;
    }
}

module.exports = OllamaClient;