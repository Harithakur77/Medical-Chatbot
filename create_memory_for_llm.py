print("Script started")
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# step :Load raw PDF(s)

DATA_PATH="data/"
def load_pdf_files(data):
    loader= DirectoryLoader(data,
                            glob='*.pdf',
                            loader_cls=PyPDFLoader)

    documents=loader.load()
    return documents

documents =load_pdf_files(data=DATA_PATH)
print("length of PDF pages:",len(documents))
print("Script finished")

# step 2: create chunks

def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size= 500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print("length of text chunks:",len(text_chunks))
