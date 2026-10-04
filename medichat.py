import os
import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CUSTOM_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know. Don't make up an answer.
Don't provide any explanations, just answer based on the context below.

Context: {context}
Question: {question}

Start the answer below:
"""


@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)


@st.cache_resource
def load_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.5,
        max_tokens=1024,
        groq_api_key=GROQ_API_KEY,
    )


def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])


def main():
    st.title("Medical Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi there! How can I assist you with your medical queries today?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Enter your query here")
    if not prompt:
        return

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY is not set. Check your .env file.")
            return

        vectorstore = get_vectorstore()

        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
        )

        response = qa_chain.invoke({"query": prompt})
        result = response["result"]
        sources = "\n\n**Sources:**\n" + "\n".join(
            f"- {d.metadata.get('source', 'unknown')} "
            f"(page {d.metadata.get('page_label', d.metadata.get('page', '?'))})"
            for d in response["source_documents"]
        )
        result_to_display = result + sources

        st.chat_message("assistant").markdown(result_to_display)
        st.session_state.messages.append({"role": "assistant", "content": result_to_display})

    except Exception as e:
        st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()