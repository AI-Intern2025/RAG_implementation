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

### 🔹 Step 1: Generate Hypothetical Scenarios
def generate_hypothetical_scenarios(query):
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "Generate different hypothetical scenarios based on the user's query."},
            {"role": "user", "content": f"Generate hypothetical situations for: {query}"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

### 🔹 Step 2: Expand Query for Web Search
def expand_query(query):
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "Expand the user query for better web search results (keep it concise)."},
            {"role": "user", "content": f"Expand this query: {query}"}
        ],
        temperature=0.5,
        max_tokens=20
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

### 🔹 Step 4: Generate Final Answer using Hypothetical Question RAG
def generate_final_answer(query):
    hypothetical_scenarios = generate_hypothetical_scenarios(query)
    expanded_query = expand_query(query)
    web_results = fetch_web_documents(expanded_query)

    # Debugging: Check retrieved web results
    print("✅ Web Search Results:", web_results)

    if "SerpAPI Error" in web_results[0]:
        return "❌ No relevant web data found. Please refine your query or check SerpAPI."

    combined_context = f"Hypothetical Scenarios: {hypothetical_scenarios}\nExpanded Query: {expanded_query}\n\n" + "\n".join(web_results)

    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "Answer user queries by generating hypothetical scenarios and using real-time web search data."},
            {"role": "user", "content": f"Query: {query}\nContext:\n{combined_context}\nProvide a well-reasoned answer:"}
        ],
        temperature=0.7,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()

### 🔹 Example Query
query = "What if artificial intelligence could develop human emotions?"
answer = generate_final_answer(query)
print("🔹 Hypothetical Question RAG Answer:", answer)
