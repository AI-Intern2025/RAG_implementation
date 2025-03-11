import os
import requests
from dotenv import load_dotenv
from groq import Groq

# Load API keys from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

### 🔹 Step 1: Classify Query into Hierarchical Categories
def classify_query(query):
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "Classify user queries into hierarchical categories like AI, Healthcare, Finance, etc."},
            {"role": "user", "content": f"Classify this query: {query}"}
        ],
        temperature=0.5,
        max_tokens=20
    )
    return response.choices[0].message.content.strip()

### 🔹 Step 2: Expand Query Using Groq (with a limit on word length)
def expand_query(query):
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "Expand user queries concisely for better web search results (keep it under 15 words)."},
            {"role": "user", "content": f"Expand this query: {query}"}
        ],
        temperature=0.5,
        max_tokens=20  # Reduce token length for better search results
    )
    return response.choices[0].message.content.strip()

### 🔹 Step 3: Fetch Web Documents using SerpAPI
def fetch_web_documents(query):
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}&num=5"
    response = requests.get(url)
    data = response.json()  # Get full API response

    print("✅ SerpAPI Raw Response:", data)  # Debugging print

    results = data.get("organic_results", [])
    web_docs = [result["snippet"] for result in results if "snippet" in result]

    if not web_docs:
        return ["SerpAPI Error: No relevant web documents found."]
    
    return web_docs

### 🔹 Step 4: Generate Final Answer using Hierarchical Index RAG
def generate_final_answer(query):
    category = classify_query(query)
    expanded_query = expand_query(query)
    web_results = fetch_web_documents(expanded_query)

    # Debugging: Check retrieved web results
    print("✅ Web Search Results:", web_results)

    if "SerpAPI Error" in web_results[0]:
        return "❌ No relevant web data found. Please refine your query or check SerpAPI."

    combined_context = f"Category: {category}\nExpanded Query: {expanded_query}\n\n" + "\n".join(web_results)

    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "Answer user queries using hierarchical topic classification and real-time web search data."},
            {"role": "user", "content": f"Query: {query}\nContext:\n{combined_context}\nProvide a detailed yet concise answer:"}
        ],
        temperature=0.5,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

### 🔹 Example Query
query = "What are the latest advancements in machine learning?"
answer = generate_final_answer(query)
print("🔹 Hierarchical RAG Answer:", answer)
