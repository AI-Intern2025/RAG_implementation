import faiss
import numpy as np
import rank_bm25
import PyPDF2
import os
import requests
from sentence_transformers import SentenceTransformer

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

groq_api_key = "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"  
groq_model = "mixtral-8x7b-32768"  

def load_pdfs(folder_path):
    documents = []
    file_names = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            with open(os.path.join(folder_path, file), "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                documents.append(text)
                file_names.append(file)
    return documents, file_names

def process_docs(folder_path):
    docs, file_names = load_pdfs(folder_path)
    if not docs:
        return None, None, None, None
    vectors = embed_model.encode(docs)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    tokenized_corpus = [doc.lower().split() for doc in docs]
    bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
    return docs, file_names, vectors, index, bm25

def summarize_with_groq(text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": groq_model,
        "messages": [
            {"role": "system", "content": "Summarize this document in a concise manner."},
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

def fusion_retrieval(query, folder_path):
    docs, file_names, vectors, index, bm25 = process_docs(folder_path)
    if not docs:
        return "No technical documents found."
    query_vec = embed_model.encode([query])
    _, I = index.search(np.array(query_vec), 3)
    retrieved_semantic = [(file_names[i], 1.0 - (_[0][i] / max(_[0]))) for i in I[0]]
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    retrieved_bm25 = [(file_names[i], bm25_scores[i]) for i in np.argsort(bm25_scores)[-3:][::-1]]
    combined_results = {}
    for doc, score in retrieved_semantic:
        combined_results[doc] = combined_results.get(doc, 0) + score * 0.6
    for doc, score in retrieved_bm25:
        combined_results[doc] = combined_results.get(doc, 0) + score * 0.4
    ranked_docs = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
    top_docs = [docs[file_names.index(doc)] for doc, _ in ranked_docs[:2]]
    summarized_text = " ".join(top_docs)[:3000]
    summary = summarize_with_groq(summarized_text)
    return summary, [doc for doc, _ in ranked_docs]

# Example Query
folder_path = r"C:\Users\shivr\OneDrive\Desktop\RAG_implementation\tech_docs"
query = "How does fine-tuning work in deep learning?"
summary, sources = fusion_retrieval(query, folder_path)
print("🔹 Fusion Retrieval Summary:", summary)
print("📂 Source Documents:", sources)