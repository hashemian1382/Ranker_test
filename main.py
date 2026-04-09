import os
import requests

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
JINA_API_KEY = os.environ.get("JINA_API_KEY")

def search_tavily(query):
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "max_results": 8
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

def main():
    queries = [
        "بهترین روش‌های یادگیری ماشین",
        "تاثیر گرمایش زمین بر اقیانوس‌ها"
    ]

    for query in queries:
        print(f"=== Query: {query} ===")
        tavily_results = search_tavily(query)

        if not tavily_results:
            print("No results from Tavily.\n")
            continue

        documents = [doc["content"] for doc in tavily_results]
        original_urls = [doc["url"] for doc in tavily_results]

        print("--- Tavily Original Ranking ---")
        for i, url in enumerate(original_urls):
            print(f"Rank {i}: {url}")

        cohere_indices = rerank_cohere(query, documents)
        print("\n--- Cohere Re-ranked ---")
        for new_rank, old_index in enumerate(cohere_indices):
            print(f"Rank {new_rank}: {original_urls[old_index]} (Original: {old_index})")

        jina_indices = rerank_jina(query, documents)
        print("\n--- Jina AI Re-ranked ---")
        for new_rank, old_index in enumerate(jina_indices):
            print(f"Rank {new_rank}: {original_urls[old_index]} (Original: {old_index})")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
