import os
from dotenv import load_dotenv
# We are importing 'download_embeddings' because that is the name in your helper.py
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# 1. Load Environment Variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set environment variables for LangChain components to pick up automatically
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

print("--- Starting Data Processing ---")

# 2. Load and Split Data
extracted_data = load_pdf_files("data")
print(f"Loaded {len(extracted_data)} documents from PDF.")

filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)
print(f"Created {len(text_chunks)} text chunks.")

# 3. Initialize Embeddings
embeddings = download_hugging_face_embeddings()
print("Embedding model loaded successfully.")

# 4. Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

# Create Index if it doesn't exist
if not pc.has_index(index_name):
    print(f"Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,     # Must match all-MiniLM-L6-v2 dimensions
        metric="cosine",   
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"Index '{index_name}' already exists.")

# 5. Push Data to Pinecone
print("Upserting vectors to Pinecone... this might take a moment.")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("--- Indexing Complete! Your data is now in Pinecone ---")