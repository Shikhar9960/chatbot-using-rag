import streamlit as st
import os
import tempfile

# Lightweight Imports (No heavy AI libraries)
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Page Config ---
st.set_page_config(page_title="Lightweight RAG Bot", layout="wide")
st.title("Light Chatbot Using RAG ")
st.caption("Uses Groq Llama 3 + BM25 (For Keyword Search)")

# --- Sidebar ---
with st.sidebar:
    api_key = st.text_input("Enter API Key", type="password")
    st.markdown("[Get Key Here](https://console.groq.com/keys)")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# --- Logic ---
def process_pdf(uploaded_file):
    # Temp file save
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    # Load
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    
    # Split (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Cleanup
    os.remove(tmp_path)
    
    # Create Retriever directly (No Vector Store needed!)
    retriever = BM25Retriever.from_documents(splits)
    retriever.k = 3  # Top 3 matching chunks layega
    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    # llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")
    # Naya aur fast model
    
    # State Management
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
        
    if uploaded_file and not st.session_state.retriever:
        with st.spinner("Processing PDF (Super Fast)..."):
            st.session_state.retriever = process_pdf(uploaded_file)
            st.success("PDF Ready!")
            
    # Chat Logic
    user_query = st.text_input("Ask your question:")
    
    if user_query and st.session_state.retriever:
        
        # Simple RAG Chain
        prompt_template = """
        Answer the question based only on the following context:
        {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        rag_chain = (
            {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        with st.spinner("Generating Answer..."):
            response = rag_chain.invoke(user_query)
            st.write("### Answer:")
            st.write(response)

else:
    st.warning("Please enter Groq API Key.")