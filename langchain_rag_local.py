"""
LangChain RAG System with Local Embeddings and Llama API
"""

import os
import json
import requests
from dotenv import load_dotenv

# Set tokenizers parallelism environment variable before importing HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from typing import Any, List, Optional

from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter


class LlamaAPILLM(BaseLLM):
    """Custom LLM for Llama API"""
    
    api_key: str
    model: str = "Llama-4-Maverick-17B-128E-Instruct-FP8"
    temperature: float = 0.7

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        url = "https://api.llama.com/v1/chat/completions"
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract the response text from Llama API format
            if 'completion_message' in data and 'content' in data['completion_message']:
                content = data['completion_message']['content']
                if isinstance(content, dict) and 'text' in content:
                    return content['text']
                elif isinstance(content, str):
                    return content
            
            # Fallback: try to extract from other possible locations
            if 'choices' in data and len(data['choices']) > 0:
                choice = data['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    return choice['message']['content']
                elif 'text' in choice:
                    return choice['text']
            
            raise ValueError(f"Unexpected response format: {data}")
            
        except Exception as e:
            raise Exception(f"API call failed: {e}")
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([{"text": text}])
        return LLMResult(generations=generations)
    
    @property
    def _llm_type(self) -> str:
        return "llama-api"


def main():
    # Load environment variables
    load_dotenv(os.getcwd()+"/.env", override=True)

    # Llama API configuration
    llama_api_key = os.getenv("LLAMA_API_KEY", "")
    
    if not llama_api_key:
        print("Error: Please set LLAMA_API_KEY in your .env file")
        print("See .env.example for the required format")
        return

    # URLs to index
    URLS_DICTIONARY = {
        "ufc_ibm_partnership": "https://newsroom.ibm.com/2024-11-14-ufc-names-ibm-as-first-ever-official-ai-partner",
        "granite.html": "https://www.ibm.com/granite",
        "products_watsonx_ai.html": "https://www.ibm.com/products/watsonx-ai",
        "products_watsonx_ai_foundation_models.html": "https://www.ibm.com/products/watsonx-ai/foundation-models",
        "watsonx_pricing.html": "https://www.ibm.com/watsonx/pricing",
        "watsonx.html": "https://www.ibm.com/watsonx",
        "products_watsonx_data.html": "https://www.ibm.com/products/watsonx-data",
        "products_watsonx_assistant.html": "https://www.ibm.com/products/watsonx-assistant",
        "products_watsonx_code_assistant.html": "https://www.ibm.com/products/watsonx-code-assistant",
        "products_watsonx_orchestrate.html": "https://www.ibm.com/products/watsonx-orchestrate",
        "products_watsonx_governance.html": "https://www.ibm.com/products/watsonx-governance",
        "granite_code_models_open_source.html": "https://research.ibm.com/blog/granite-code-models-open-source",
        "red_hat_enterprise_linux_ai.html": "https://www.redhat.com/en/about/press-releases/red-hat-delivers-accessible-open-source-generative-ai-innovation-red-hat-enterprise-linux-ai",
        "model_choice.html": "https://www.ibm.com/blog/announcement/enterprise-grade-model-choices/",
        "democratizing.html": "https://www.ibm.com/blog/announcement/democratizing-large-language-model-development-with-instructlab-support-in-watsonx-ai/",
        "ibm_consulting_expands_ai.html": "https://newsroom.ibm.com/Blog-IBM-Consulting-Expands-Capabilities-to-Help-Enterprises-Scale-AI",
        "ibm_data_product_hub.html": "https://www.ibm.com/products/data-product-hub",
        "ibm_price_performance_data.html": "https://www.ibm.com/blog/announcement/delivering-superior-price-performance-and-enhanced-data-management-for-ai-with-ibm-watsonx-data/",
        "ibm_bi_adoption.html": "https://www.ibm.com/blog/a-new-era-in-bi-overcoming-low-adoption-to-make-smart-decisions-accessible-for-all/",
        "code_assistant_for_java.html": "https://www.ibm.com/blog/announcement/watsonx-code-assistant-java/",
        "accelerating_gen_ai.html": "https://newsroom.ibm.com/Blog-How-IBM-Cloud-is-Accelerating-Business-Outcomes-with-Gen-AI",
        "watsonx_open_source.html": "https://newsroom.ibm.com/2024-05-21-IBM-Unveils-Next-Chapter-of-watsonx-with-Open-Source,-Product-Ecosystem-Innovations-to-Drive-Enterprise-AI-at-Scale",
        "ibm_concert.html": "https://www.ibm.com/products/concert",
        "ibm_consulting_advantage_news.html": "https://newsroom.ibm.com/2024-01-17-IBM-Introduces-IBM-Consulting-Advantage,-an-AI-Services-Platform-and-Library-of-Assistants-to-Empower-Consultants",
        "ibm_consulting_advantage_info.html": "https://www.ibm.com/consulting/info/ibm-consulting-advantage"
}

    print("Loading documents from URLs...")
    documents = []

    for url in list(URLS_DICTIONARY.values()):
        try:
            print(f"Attempting to load: {url}")
            loader = WebBaseLoader(url)
            data = loader.load()
            documents += data
            print(f"✓ Successfully loaded: {url}")
        except Exception as e:
            print(f"✗ Error loading {url}: {e}")

    # Clean up documents
    for doc in documents:
        doc.page_content = " ".join(doc.page_content.split())  # remove white space

    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Initialize local embeddings (no API key needed)
    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector store
    print("Creating vector store...")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    # Set up retriever
    retriever = vectorstore.as_retriever()

    # Set up LLM with Llama API
    print("Initializing LLM...")
    llm = LlamaAPILLM(api_key=llama_api_key)

    # Set up prompt template
    template = """Generate a summary of the context that answers the question. Explain the answer in multiple steps if possible. 
    Answer style should match the context. Ideal Answer Length 2-3 sentences.\n\n{context}\nQuestion: {question}\nAnswer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # Create RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\nRAG system ready! You can now ask questions about IBM products and technologies.")
    print("Example questions:")
    print("- Tell me about the UFC announcement from November 14, 2024")
    print("- What is watsonx.data?")
    print("- What does watsonx.ai do?")
    print("\nType 'quit' to exit.\n")

    # Interactive question loop
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if question:
            try:
                print("\nThinking...")
                
                # Debug: Show retrieved context
                retrieved_docs = retriever.invoke(question)
                print(f"\nRetrieved {len(retrieved_docs)} document(s) from vector store:")
                for i, doc in enumerate(retrieved_docs):
                    print(f"\n--- Document {i+1} ---")
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    print(f"Content preview: {doc.page_content[:200]}...")
                
                # Get the full response
                response = rag_chain.invoke(question)
                if response:
                    print(f"\nAnswer: {response}")
                else:
                    print("Error: Received empty response from LLM")
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()