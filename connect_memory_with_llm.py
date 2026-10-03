import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS


# Step 1: LLM
HF_TOKEN = os.getenv("HF_TOKEN")
#HUGGINGFACE_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"   # gated: accept the license on its HF page first

HUGGINGFACE_REPO_ID = "openai/gpt-oss-20b"

def load_llm(repo_id):
    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",
        provider="groq",          # if this fails, try "together" or "cerebras"
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN,
    )
    return ChatHuggingFace(llm=endpoint)

# Step 2: Prompt
CUSTOM_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know. Don't make up an answer.
Don't provide any explanations, just answer based on the context below.

Context: {context}
Question: {question}

Start the answer below:
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
    llm=load_llm(HUGGINGFACE_REPO_ID),
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
    print(f"- {d.metadata['source']}, page {d.metadata['page_label']}")