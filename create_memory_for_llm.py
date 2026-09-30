print("Script started")
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#step `:Load raw PDF(s)

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