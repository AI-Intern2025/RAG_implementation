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

### 🔹 Step 1: Expand Query using Groq
def expand_query_groq(query):
    print(f"🔍 Expanding query: {query}")

    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "Expand user queries for better search results."},
            {"role": "user", "content": f"Expand this query: {query}"}
        ],
        temperature=0.7,
        max_tokens=50
    )

    expanded_query = response.choices[0].message.content.strip()
    
    # Truncate expanded query to avoid issues with long search strings
    expanded_query = expanded_query[:100]  
    print(f"✅ Expanded Query: {expanded_query}")
    
    return expanded_query

### 🔹 Step 2: Fetch Web Documents using SerpAPI
def fetch_web_documents(query):
    print(f"🔍 Searching SerpAPI for: {query}")

    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}&num=5"
    response = requests.get(url).json()
    
    # Debugging: Print raw response
    print(f"✅ SerpAPI Raw Response: {response}")

    if "error" in response:
        print(f"❌ SerpAPI Error: {response['error']}")
        return ["SerpAPI Error: No relevant web documents found."]

    web_docs = [result["snippet"] for result in response.get("organic_results", []) if "snippet" in result]

    if not web_docs:
        print("⚠️ No relevant web documents found. Retrying with simpler query...")
        return fetch_web_documents("latest AI trends")  # Try a fallback search

    return web_docs

### 🔹 Step 3: Combine Expanded Query + Web Documents
def get_combined_context(query):
    expanded_query = expand_query_groq(query)
    web_results = fetch_web_documents(expanded_query)

    combined_context = "\n".join(web_results)
    print(f"✅ Final Web Search Context: {combined_context[:300]}...")  # Show only first 300 chars

    return combined_context

### 🔹 Step 4: Generate Final Answer using Groq
def generate_final_answer(query):
    context = get_combined_context(query)
    
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "Answer user queries using real-time web search data."},
            {"role": "user", "content": f"Query: {query}\nContext:\n{context}\nProvide a detailed yet concise answer:"}
        ],
        temperature=0.5,
        max_tokens=200
    )

    final_answer = response.choices[0].message.content.strip()
    print(f"🔹 Final Answer: {final_answer}")

    return final_answer

### 🔹 Example Query
query = "What are the latest advancements in artificial intelligence?"
answer = generate_final_answer(query)

print("\n🔹 Query Expansion RAG Answer:", answer)

