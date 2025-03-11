import os
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
from groq import Groq

# Load API keys from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Ensure nltk dependencies are downloaded
nltk.download("punkt")

### 🔹 Step 1: Semantic Chunking
def semantic_chunking(text):
    """Splits text into semantically meaningful chunks."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        
        # Ask Groq if we should split here
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "Decide if a chunk should end after this sentence."},
                {"role": "user", "content": f"Sentence: {sentence}\nShould we split here? Answer 'yes' or 'no'."}
            ],
            temperature=0.3,
            max_tokens=5
        )
        
        decision = response.choices[0].message.content.strip().lower()
        if decision == "yes":
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

### 🔹 Step 2: Retrieve Relevant Chunks
def retrieve_relevant_chunks(query, chunks):
    """Finds the most relevant chunk based on query similarity."""
    best_chunk = ""
    best_score = 0

    for chunk in chunks:
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "Rate the relevance of this chunk to the query on a scale of 0 to 100."},
                {"role": "user", "content": f"Query: {query}\nChunk: {chunk}\nRelevance score:"}
            ],
            temperature=0.3,
            max_tokens=5
        )

        try:
            score = int(response.choices[0].message.content.strip())
            if score > best_score:
                best_score = score
                best_chunk = chunk
        except ValueError:
            continue  # Skip invalid responses

    return best_chunk

### 🔹 Step 3: Answer Query Using RAG
def process_query_with_rag(query, document):
    chunks = semantic_chunking(document)
    best_chunk = retrieve_relevant_chunks(query, chunks)

    print(f"✅ Total Chunks Created: {len(chunks)}")
    print(f"✅ Best Matching Chunk: {best_chunk[:150]}...")

    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "Answer the query using the retrieved chunk."},
            {"role": "user", "content": f"Query: {query}\nRelevant Chunk: {best_chunk}\nAnswer:"}
        ],
        temperature=0.7,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()

### 🔹 Example Query
query = "What are the ethical concerns in AI?"
document = """Artificial Intelligence (AI) has transformed multiple industries. However, ethical concerns persist. Bias in AI models 
leads to unfair decisions. Privacy issues arise due to large-scale data collection. Transparency is another challenge—AI models 
often act as black boxes. AI in automation raises employment concerns. Regulations are being introduced to ensure fairness and accountability."""

answer = process_query_with_rag(query, document)
print("🔹 Semantic Chunking RAG Answer:", answer)
