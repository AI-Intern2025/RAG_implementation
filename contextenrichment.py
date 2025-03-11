import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import requests


GROQ_API_KEY = "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"  
GROQ_MODEL = "mixtral-8x7b-32768"  # Choose an available Groq model

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample document database (Primary Knowledge)
documents = [
    "Artificial Intelligence is transforming industries.",
    "Machine Learning is a subset of AI that learns from data.",
    "Neural Networks are the backbone of Deep Learning.",
    "Data preprocessing is crucial for machine learning accuracy.",
    "Transfer learning allows models to adapt quickly to new tasks."
]

# Extra knowledge for context enrichment
extra_knowledge = {
    "machine learning": "Machine learning uses algorithms to analyze data, learn patterns, and make decisions.",
    "artificial intelligence": "AI refers to computer systems that can perform tasks typically requiring human intelligence."
}

# Create FAISS index for document retrieval
vectors = embed_model.encode(documents)
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))

# ✅ Function to Query ChatGroq
def chatgroq_answer(query, context):
    url = "https://api.groq.com/openai/v1/chat/completions"  # ✅ Correct API Endpoint
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "Answer the question using the provided context."},
            {"role": "user", "content": f"Question: {query}\nContext: {context}"}
        ],
        "max_tokens": 150
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response_json = response.json()
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "No response generated.")
    except Exception as e:
        print(f"❌ Groq API Error: {e}")
        return "Groq API failed to generate an answer."

# ✅ Context Enrichment RAG Function
def context_enrichment_rag(query):
    # Retrieve top document from FAISS
    query_vec = embed_model.encode([query])
    _, I = index.search(np.array(query_vec), 1)
    retrieved_doc = documents[I[0][0]]

    # Enrich context with extra knowledge
    enriched_context = retrieved_doc  # Start with retrieved document
    for keyword, extra_info in extra_knowledge.items():
        if keyword in query.lower():  # If query contains a keyword, add extra info
            enriched_context += " " + extra_info

    # Get Answer from ChatGroq
    response = chatgroq_answer(query, enriched_context)
    
    return response

# Example query
query = "What is llm?"
print("🔹 Context Enrichment RAG Answer:", context_enrichment_rag(query))
