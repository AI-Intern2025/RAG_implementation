import os
import tiktoken
from dotenv import load_dotenv
from groq import Groq

# Load API keys from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

# Tokenizer for counting tokens (Using GPT-4 encoding for estimation)
enc = tiktoken.encoding_for_model("gpt-4")

### 🔹 Step 1: Adaptive Chunk Size Selection
def choose_chunk_size(text):
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "Analyze the given text and determine the optimal chunk size for retrieval-augmented generation."},
            {"role": "user", "content": f"Analyze this text and suggest the best chunk size: {text[:1000]} ..."}
        ],
        temperature=0.5,
        max_tokens=50
    )
    suggested_size = response.choices[0].message.content.strip()
    
    try:
        chunk_size = int(suggested_size.split()[0])  # Extract first number
        return chunk_size if chunk_size > 50 else 100  # Ensure reasonable size
    except ValueError:
        return 200  # Default to 200 tokens if Groq fails to return a number

### 🔹 Step 2: Chunk the Document
def chunk_document(text, chunk_size):
    tokens = enc.encode(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [enc.decode(chunk) for chunk in chunks]

### 🔹 Step 3: Process Query with RAG
def process_query_with_rag(query, document):
    chunk_size = choose_chunk_size(document)
    chunks = chunk_document(document, chunk_size)
    
    print(f"✅ Chosen Chunk Size: {chunk_size} tokens")
    print(f"✅ Total Chunks Created: {len(chunks)}")
    
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "Use the provided document chunks to answer the user's query."},
            {"role": "user", "content": f"Query: {query}\nDocument Chunks:\n{chunks[:3]}...\nProvide a response:"}
        ],
        temperature=0.7,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()

### 🔹 Example Query
query = "What are the key challenges in AI research?"
document = """Artificial Intelligence (AI) research has rapidly evolved, facing various technical and ethical challenges.
The complexity of deep learning models, data privacy concerns, and bias in machine learning algorithms are major areas of concern.
Additionally, AI alignment, transparency, and interpretability remain unsolved problems in the field.
Recent research aims to improve the efficiency of AI models while ensuring fairness and accountability.
Advancements in reinforcement learning and transfer learning also present new challenges in scalability and real-world application.
The future of AI research depends on balancing innovation with ethical considerations."""

answer = process_query_with_rag(query, document)
print("🔹 Choose Chunk Size RAG Answer:", answer)
