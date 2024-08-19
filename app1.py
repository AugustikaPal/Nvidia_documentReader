
import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the Streamlit UI
st.set_page_config(page_title="NVIDIA NIM Demo", layout="wide")
st.title("NVIDIA NIM Demo")

# Function to perform vector embedding
def vector_embedding():
    if "vectors" not in st.session_state:
        if not os.getenv("NVIDIA_API_KEY"):
            st.error("API key is required to perform embedding.")
            return
        st.session_state.embeddings = NVIDIAEmbeddings(api_key=os.getenv("NVIDIA_API_KEY"))
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings

# Define the language model and prompt
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
)

# Main layout with columns
col1, col2 = st.columns([2, 1])

with col1:
    st.header("API Key Setup")
    
    # API Key Input
    api_key = st.text_input("Enter your NVIDIA API Key", type="password")
    if api_key:
        os.environ['NVIDIA_API_KEY'] = api_key
        st.success("API key has been set.")

    st.header("Document Embedding")
    if st.button("Load and Embed Documents"):
        if not os.getenv("NVIDIA_API_KEY"):
            st.error("API key is required. Please enter it above.")
        else:
            with st.spinner("Loading and embedding documents..."):
                vector_embedding()
                st.success("Vector Store DB is ready!")

with col2:
    st.header("Ask a Question")
    
    prompt1 = st.text_input("Enter your question from documents")
    if st.button("Submit Question"):
        if 'vectors' in st.session_state:
            with st.spinner("Processing your question..."):
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                response_time = time.process_time() - start
                
                st.write(f"Response time: {response_time:.2f} seconds")
                st.write("Answer:", response.get('answer', 'No answer found.'))

                with st.expander("Document Similarity Search"):
                    for i, doc in enumerate(response.get("context", [])):
                        st.write(f"Document {i+1}:")
                        st.write(doc.page_content)
                        st.write("--------------------------------")
        else:
            st.error("Please load documents first.")
