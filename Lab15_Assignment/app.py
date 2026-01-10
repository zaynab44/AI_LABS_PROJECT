import streamlit as st

from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -----------------------------
# Load Documents
# -----------------------------
loader = TextLoader("data/sample_docs.txt")
documents = loader.load()

# -----------------------------
# Split Documents
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# -----------------------------
# Create Embeddings & Vector Store
# -----------------------------
embeddings = OllamaEmbeddings(model="llama2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# -----------------------------
# Prompt for RAG
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using only the given context."),
        ("user", "Context:\n{context}\n\nQuestion:\n{question}")
    ]
)

# -----------------------------
# LLM & RAG Chain
# -----------------------------
llm = Ollama(model="llama2")
output_parser = StrOutputParser()

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | output_parser
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 RAG Chatbot using Ollama & LangChain")

user_question = st.text_input("Ask a question based on the document:")

if user_question:
    answer = rag_chain.invoke(user_question)
    st.write("### Answer:")
    st.write(answer)