import os
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# Load the pdf sample data

def load_pdf_file(data):
    loader=DirectoryLoader(data,
                           glob='*.pdf',
                           loader_cls=PyPDFLoader)
    documents=loader.load()

    return  documents


# Splitting the data into Text Chunks

def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks


# Download an Embedding model from Hugging Face


def download_hugging_face_embeddings():
    embeddings=HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings



