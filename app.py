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
    try:
        response = requests.post(url, json=payload)
        return response.json().get("results", [])
    except:
        return []

def rerank_cohere(query, documents):
    url = "https://api.cohere.ai/v1/rerank"
    headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "rerank-multilingual-v3.0", "query": query, "documents": documents, "return_documents": False}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return [res["index"] for res in response.json().get("results", [])]
    except:
        return list(range(len(documents)))

def rerank_jina(query, documents):
    url = "https://api.jina.ai/v1/rerank"
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "jina-reranker-v2-base-multilingual", "query": query, "documents": documents}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return [res["index"] for res in response.json().get("results", [])]
    except:
        return list(range(len(documents)))

# تابع محاسبه ضریب همبستگی اسپیرمن
def calculate_spearman(list1, list2):
    n = len(list1)
    if n < 2: return 1.0
    
    d_squared_sum = 0
    for i, item in enumerate(list1):
        if item in list2:
            rank1 = i + 1
            rank2 = list2.index(item) + 1
            d_squared_sum += (rank1 - rank2) ** 2
            
    rho = 1 - ((6 * d_squared_sum) / (n * (n**2 - 1)))
    return rho

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def handle_search():
    query = request.json.get('query')
    if not query: return jsonify({"error": "Query is required"}), 400

    tavily_results = search_tavily(query)
    if not tavily_results: return jsonify({"tavily": [], "cohere": [], "jina": []})

    documents = [doc["content"] for doc in tavily_results]
    tavily_data = [{"rank": i+1, "title": doc["title"], "url": doc["url"]} for i, doc in enumerate(tavily_results)]

    cohere_indices = rerank_cohere(query, documents)
    cohere_data = [{"rank": new_rank+1, "original_rank": old_index+1, "title": tavily_results[old_index]["title"], "url": tavily_results[old_index]["url"]} for new_rank, old_index in enumerate(cohere_indices)]

    jina_indices = rerank_jina(query, documents)
    jina_data = [{"rank": new_rank+1, "original_rank": old_index+1, "title": tavily_results[old_index]["title"], "url": tavily_results[old_index]["url"]} for new_rank, old_index in enumerate(jina_indices)]

    return jsonify({"tavily": tavily_data, "cohere": cohere_data, "jina": jina_data})

@app.route('/evaluate', methods=['POST'])
def handle_evaluate():
    queries = request.json.get('queries', [])
    if not queries: return jsonify({"error": "No queries provided"}), 400

    metrics = {
        "tavily_cohere": [],
        "tavily_jina": [],
        "cohere_jina": []
    }

    processed_queries = 0

    for query in queries:
        query = query.strip()
        if not query: continue
        
        tavily_results = search_tavily(query)
        if not tavily_results or len(tavily_results) < 2: continue

        documents = [doc["content"] for doc in tavily_results]
        tavily_urls = [doc["url"] for doc in tavily_results]

        cohere_indices = rerank_cohere(query, documents)
        cohere_urls = [tavily_urls[i] for i in cohere_indices]

        jina_indices = rerank_jina(query, documents)
        jina_urls = [tavily_urls[i] for i in jina_indices]

        # محاسبه اسپیرمن برای این کوئری
        metrics["tavily_cohere"].append(calculate_spearman(tavily_urls, cohere_urls))
        metrics["tavily_jina"].append(calculate_spearman(tavily_urls, jina_urls))
        metrics["cohere_jina"].append(calculate_spearman(cohere_urls, jina_urls))
        
        processed_queries += 1

    if processed_queries == 0:
        return jsonify({"error": "Failed to process queries"}), 400

    # محاسبه میانگین کل
    avg_tc = sum(metrics["tavily_cohere"]) / len(metrics["tavily_cohere"])
    avg_tj = sum(metrics["tavily_jina"]) / len(metrics["tavily_jina"])
    avg_cj = sum(metrics["cohere_jina"]) / len(metrics["cohere_jina"])

    return jsonify({
        "processed_count": processed_queries,
        "avg_tavily_cohere": round(avg_tc, 3),
        "avg_tavily_jina": round(avg_tj, 3),
        "avg_cohere_jina": round(avg_cj, 3)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
