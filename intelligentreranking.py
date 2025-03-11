import requests
from bs4 import BeautifulSoup
from googlesearch import search
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import json
import time


GROQ_API_KEY = "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" 
GROQ_MODEL = "mixtral-8x7b-32768"  

#  Function to fetch top AI-related articles with retries
def fetch_articles(query, num_results=5):
    articles = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for url in search(query, num_results=num_results):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()  # Raise HTTP errors if any
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text() for p in paragraphs])
            if text.strip():  # Ensure it's not empty
                articles.append((url, text))
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to fetch {url}: {e}")
        time.sleep(1)  # Prevent rate limiting

    return articles

#  Function to compute BM25 & cosine similarity rankings
def rank_articles(articles, query):
    if not articles:
        return []

    tokenized_docs = [article[1].split() for article in articles]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = np.array(bm25.get_scores(query.split()))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode([article[1] for article in articles], convert_to_tensor=True)

    cosine_scores = np.array([util.pytorch_cos_sim(query_embedding, doc_embedding).item() for doc_embedding in doc_embeddings])

    #  Normalize scores (avoid divide by zero)
    epsilon = 1e-8
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)
    cosine_scores = (cosine_scores - np.min(cosine_scores)) / (np.max(cosine_scores) - np.min(cosine_scores) + epsilon)

    #  Fusion Ranking: Weighted Sum
    final_scores = (0.6 * bm25_scores) + (0.4 * cosine_scores)
    sorted_indices = np.argsort(final_scores)[::-1]

    return [(articles[i][0], articles[i][1], final_scores[i]) for i in sorted_indices]

#  Function to summarize content using Groq API
def summarize_with_groq(text):
    url = "https://api.groq.com/openai/v1/chat/completions"  # ✅ Correct API Endpoint
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "Summarize this article in a concise manner."},
            {"role": "user", "content": text[:3000]}  # Limit input to avoid token overflow
        ],
        "max_tokens": 200
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response_json = response.json()
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "No summary available.")
    except Exception as e:
        print(f"❌ Groq API Error: {e}")
        return "Groq API failed to generate a summary."

#  Main Execution
if __name__ == "__main__":
    query = "Latest advancements in AI technology"
    print(f"🔎 Searching for: {query}...\n")

    articles = fetch_articles(query)

    if not articles:
        print("❌ No articles found. Exiting.")
        exit()

    ranked_articles = rank_articles(articles, query)

    if not ranked_articles:
        print("❌ No relevant articles found. Exiting.")
        exit()

    top_article_url, top_article_text, _ = ranked_articles[0]

    print(f"\n🌐 Most Relevant Article: {top_article_url}\n")

    summary = summarize_with_groq(top_article_text)

    print("📝 Summary:")
    print(summary)
