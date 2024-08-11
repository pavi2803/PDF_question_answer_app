# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:27:15 2024

@author: Pavithra
"""

import os
import streamlit as st
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API keys
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# Set the title for the Streamlit app
st.title("PDF Q&A App using Gemma")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7B-it")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate responses based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to process the PDF and create a vector store
def process_pdf_and_create_vectors(uploaded_file):
    # Initialize the PDF reader and extract text from the PDF
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Store the document content in session state
    st.session_state.docs = [{"page_content": text, "metadata": {}}]

    # Split the document into chunks for processing
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    # Initialize the embeddings and create the vector store
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# File uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Process the uploaded PDF if a file is provided
if uploaded_file is not None:
    process_pdf_and_create_vectors(uploaded_file)
    st.write("PDF processed and vector store created.")

# Text input for the question
prompt1 = st.text_input("What do you want to know from the document?")

# Handle the question input and provide an answer based on the PDF content
if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input': prompt1})
    st.write(response['answer'])
