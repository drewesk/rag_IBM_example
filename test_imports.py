#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import os
from dotenv import load_dotenv

# Test all imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("✅ All imports successful!")

# Test environment variables
load_dotenv(os.getcwd()+"/.env", override=True)
llama_api_key = os.getenv("LLAMA_API_KEY", "")

if llama_api_key:
    print("✅ LLAMA_API_KEY found in .env file")
else:
    print("❌ LLAMA_API_KEY not found in .env file")

print("\nReady to run langchain_rag_local.py!")