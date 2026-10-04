import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

load_dotenv()

# Step 1: LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not GROQ_API_KEY:
    raise SystemExit("GROQ_API_KEY is not set. Check your .env file.")

def load_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.5,
        max_tokens=1024,
        groq_api_key=GROQ_API_KEY,
    )

# Step 2: Prompt
CUSTOM_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
Give a clear, detailed answer in several sentences or bullet points, covering definition, causes, types, symptoms and treatment if they appear in the context.
If the context doesn't contain the answer, just say that you don't know. Don't make up an answer.

Context: {context}
Question: {question}

Detailed answer:
"""

def set_custom_prompt_template():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

# Step 3: Load vector DB
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt_template()},
)

user_query = input("Enter your query: ")
response = qa_chain.invoke({"query": user_query})
print("Answer:", response["result"])
print("\nSources:")
for d in response["source_documents"]:
    page = d.metadata.get("page_label", d.metadata.get("page", "?"))
    print(f"- {d.metadata.get('source', 'unknown')}, page {page}")