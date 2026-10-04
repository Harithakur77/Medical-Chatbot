import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Step 1: Load PDFs
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

documents = load_pdf_files(DATA_PATH)
print("Length of PDF pages:", len(documents))
if not documents:
    raise SystemExit("No PDFs found in data/. Add a PDF and run again.")

# Step 2: Create chunks
def create_chunks(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(extracted_data)

text_chunks = create_chunks(documents)
print("Length of text chunks:", len(text_chunks))

# Step 3: Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Store in FAISS
os.makedirs("vectorstore", exist_ok=True)
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
print("Vectorstore saved to", DB_FAISS_PATH)