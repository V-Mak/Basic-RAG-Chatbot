from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================
# 1. Load PDF
# =========================

loader = PyPDFLoader("data/Python for Probability, Statistics, and Machine Learning.pdf")

documents = loader.load()

print("Number of documents:", len(documents))


# =========================
# 2. Split Documents
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# =========================
# 3. Create Embedding Model
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# 4. Create FAISS Vector Store
# =========================

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS index created!")


# =========================
# 5. Save FAISS Index
# =========================

vector_store.save_local("vector_store")

print("FAISS index saved successfully!")