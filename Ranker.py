import os
import requests

def get_cohere_indices(query, documents):
    api_key = os.environ.get("COHERE_API_KEY")
    url = "https://api.cohere.ai/v1/rerank"
    headers = {
        "Authorization": f"Bearer {api_key}", 
        "Content-Type": "application/json"
    }
    payload = {
        "model": "rerank-multilingual-v3.0", 
        "query": query, 
        "documents": documents, 
        "return_documents": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return [res["index"] for res in response.json().get("results", [])]
    except:
        return list(range(len(documents)))

def get_jina_indices(query, documents):
    api_key = os.environ.get("JINA_API_KEY")
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Authorization": f"Bearer {api_key}", 
        "Content-Type": "application/json"
    }
    payload = {
        "model": "jina-reranker-v2-base-multilingual", 
        "query": query, 
        "documents": documents
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return [res["index"] for res in response.json().get("results", [])]
    except:
        return list(range(len(documents)))

def rerank_results(query, results, method="cohere"):
    if not results:
        return []

    documents = [doc.get("content", "") for doc in results]
    
    if method == "cohere":
        indices = get_cohere_indices(query, documents)
        
    elif method == "jina":
        indices = get_jina_indices(query, documents)
        
    elif method == "mix":
        cohere_indices = get_cohere_indices(query, documents)
        jina_indices = get_jina_indices(query, documents)
        
        doc_count = len(documents)
        ranks = {i: {"cohere": doc_count, "jina": doc_count} for i in range(doc_count)}
        
        for rank, orig_idx in enumerate(cohere_indices):
            ranks[orig_idx]["cohere"] = rank
            
        for rank, orig_idx in enumerate(jina_indices):
            ranks[orig_idx]["jina"] = rank
            
        indices = sorted(ranks.keys(), key=lambda i: (ranks[i]["cohere"] + ranks[i]["jina"]) / 2.0)
        
    else:
        return results

    reranked_results = [results[i] for i in indices]
    
    return reranked_results