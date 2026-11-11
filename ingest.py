import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import Distance

load_dotenv()

COLLECTION_NAME = "techno_products"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("Step 1: Loading document...")

possible_paths = [
    "Data/Sample_Catalog",
    "data/sample_catalog.txt",
    "data/Sample_Catalog.txt",
    "Sample_Catalog"
]

file_path = next((p for p in possible_paths if os.path.isfile(p)), None)

if not file_path:
    print("No catalog found. Creating sample catalog...")
    os.makedirs("data", exist_ok=True)

    file_path = "data/sample_catalog.txt"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(
            "M30 Bolt Rs 85 Stock 5000\n"
            "Bearing 6205 Rs 450\n"
            "Motor 2HP Rs 12500\n"
            "Motor sound issue may be caused by bearing problems.\n"
        )

docs = TextLoader(file_path, encoding="utf-8").load()
print(f"Loaded {len(docs)} documents")

print("Step 2: Splitting documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

print("Step 3: Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Step 4: Connecting to Qdrant...")

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

if not url or not api_key:
    raise ValueError("QDRANT_URL or QDRANT_API_KEY missing in .env")

client = QdrantClient(
    url=url,
    api_key=api_key
)

# WARNING: This deletes the existing collection and its data.
# Use this only when you intentionally want to rebuild the collection.
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

print("Step 5: Creating Qdrant vector store...")

vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url=url,
    api_key=api_key,
    collection_name=COLLECTION_NAME,
    distance=Distance.COSINE,
)

print(f"\nSUCCESS! {len(chunks)} chunks uploaded to Qdrant.")
print(f"Collection: {COLLECTION_NAME}")
print("Next: streamlit run app.py")
