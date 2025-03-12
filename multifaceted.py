import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import datetime
import requests
import json

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

groq_api_key = ""
groq_model = "mixtral-8x7b-32768"

# Sample document database (With categories and timestamps)
documents = [
    {"text": "Artificial Intelligence is transforming industries.", "category": "AI", "date": "2022-01-10"},
    {"text": "Machine Learning is a subset of AI that learns from data.", "category": "ML", "date": "2023-06-15"},
    {"text": "Neural Networks are the backbone of Deep Learning.", "category": "Deep Learning", "date": "2021-09-05"},
    {"text": "Data preprocessing is crucial for machine learning accuracy.", "category": "ML", "date": "2024-02-01"},
    {"text": "Transfer learning allows models to adapt quickly to new tasks.", "category": "ML", "date": "2023-12-10"}
]

# Convert text to embeddings
texts = [doc["text"] for doc in documents]
vectors = embed_model.encode(texts)

# Create FAISS index for retrieval
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))

# ✅ Function to call Groq API for answer generation
def query_groq(context, query):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": groq_model,
        "messages": [
            {"role": "system", "content": "Answer the question based on the given context."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {query}"}
        ],
        "max_tokens": 200
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response_json = response.json()
    return response_json.get("choices", [{}])[0].get("message", {}).get("content", "No response available.")

# ✅ Multifaceted RAG Function
def multifaceted_rag(query, preferred_category="ML", recent_years=2):
    # Step 1: Retrieve top 3 matching documents
    query_vec = embed_model.encode([query])
    _, I = index.search(np.array(query_vec), 3)
    retrieved_docs = [documents[i] for i in I[0]]
    
    # Step 2: Apply Multifaceted Filtering
    filtered_docs = []
    current_year = datetime.datetime.now().year
    
    for doc in retrieved_docs:
        doc_year = int(doc["date"].split("-")[0])
        if doc["category"] == preferred_category and (current_year - doc_year) <= recent_years:
            filtered_docs.append(doc["text"])  # Keep only relevant documents

    # Step 3: If no filtered docs, fall back to general retrieval
    final_context = " ".join(filtered_docs) if filtered_docs else retrieved_docs[0]["text"]

    # Step 4: Query ChatGroq for answer
    response = query_groq(final_context, query)
    
    return response

# Example query
query = "What is machine learning?"
print("Multifaceted RAG Answer:", multifaceted_rag(query))
