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
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings


from dotenv import load_dotenv


load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")


st.title("PDF Q&A app using Gemma")

llm = ChatGroq(groq_api_key=groq_api_key,model_name="Gemma-7B-it")

print(llm)


prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate responses based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """)
    
    

def vector_embedding():
    
    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            with open("uploaded_temp.pdf", "wb") as f:
                f.write(uploaded_file.read())
            
            # Initialize your loader with the temporary file path
            st.session_state.loader = PyPDFDirectoryLoader("uploaded_temp.pdf")
            st.session_state.docs = st.session_state.loader.load()
        @st.session_state.loader=PyPDFDirectoryLoader("./pdfs")
        @st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,chunk_overlap=128)
        
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        


if st.button("Fetch all data from PDF"):
    vector_embedding()
    st.write("Vector Store DB is ready!")

        

prompt1 = st.text_input("What do you wanna know from the document?"
                    )


    
    
import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retreiver = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retreiver,document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input':prompt1})
    st.write(response['answer'])
    
    
        
        
        
    
