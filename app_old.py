import os
import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
JINA_API_KEY = os.environ.get("JINA_API_KEY")

def search_tavily(query):
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "max_results": 10
    }
    response = requests.post(url, json=payload)
    return response.json().get("results", [])

def rerank_cohere(query, documents):
    url = "https://api.cohere.ai/v1/rerank"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "rerank-multilingual-v3.0",
        "query": query,
        "documents": documents,
        "return_documents": False
    }
    response = requests.post(url, headers=headers, json=payload)
    results = response.json().get("results", [])
    return [res["index"] for res in results]

def rerank_jina(query, documents):
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "documents": documents
    }
    response = requests.post(url, headers=headers, json=payload)
    results = response.json().get("results", [])
    return [res["index"] for res in results]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def handle_search():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # 1. Search with Tavily
    tavily_results = search_tavily(query)
    if not tavily_results:
        return jsonify({"tavily": [], "cohere": [], "jina": []})

    documents = [doc["content"] for doc in tavily_results]
    
    # Base Tavily Data
    tavily_data = [{"rank": i+1, "title": doc["title"], "url": doc["url"]} for i, doc in enumerate(tavily_results)]

    # 2. Re-rank with Cohere
    cohere_indices = rerank_cohere(query, documents)
    cohere_data = [
        {"rank": new_rank+1, "original_rank": old_index+1, "title": tavily_results[old_index]["title"], "url": tavily_results[old_index]["url"]}
        for new_rank, old_index in enumerate(cohere_indices)
    ]

    # 3. Re-rank with Jina
    jina_indices = rerank_jina(query, documents)
    jina_data = [
        {"rank": new_rank+1, "original_rank": old_index+1, "title": tavily_results[old_index]["title"], "url": tavily_results[old_index]["url"]}
        for new_rank, old_index in enumerate(jina_indices)
    ]

    return jsonify({
        "tavily": tavily_data,
        "cohere": cohere_data,
        "jina": jina_data
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
