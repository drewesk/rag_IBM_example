# LangChain RAG System with Local Embeddings + Llama API

This project implements a Retrieval Augmented Generation (RAG) system using LangChain with local embeddings and Meta-Llama API to answer questions about IBM products and technologies.

## Key Features

- **Local Embeddings**: Uses HuggingFace embeddings locally
- **Llama API**: Only LLM calls go to external API
- **Cost-Effective**: Reduces API costs by processing embeddings locally
- **Vector Database**: Stores embeddings in Chroma vector database

## Prerequisites

1. **Llama API Key**: You need a Llama API key

## Setup Instructions

### 1. Install Dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Configure Credentials

Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

Edit the `.env` file and add your credentials:

```
LLAMA_API_KEY=your_llama_api_key_here
```

### 3. Run the Application

```bash
python langchain_rag_local.py
```

## How It Works

1. **Document Loading**: Fetches content from 25 IBM-related webpages
2. **Text Processing**: Cleans and splits documents into manageable chunks
3. **Vector Embeddings**: Uses HuggingFace embeddings locally to create vector representations
4. **Vector Store**: Stores embeddings in Chroma vector database
5. **Retrieval**: Finds relevant documents based on user questions
6. **Generation**: Uses Llama API to generate answers based on retrieved context

## Example Questions

- "Tell me about the UFC announcement from November 14, 2024"
- "What is watsonx.data?"
- "What does watsonx.ai do?"
- "Tell me about IBM Granite models"

## Files

- `langchain_rag_local.py` - Main application script
- `requirements.txt` - Python dependencies
- `.env.example` - Template for environment variables
- `README.md` - This file
