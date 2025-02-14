# # from langchain_community.vectorstores import Chroma
# # from langchain_huggingface import HuggingFaceEmbeddings
# # import os
# # from langchain_groq import ChatGroq
# # from scrape import crawl_domain

# # groq_api_key = os.getenv("GROQ_API_KEY")
# # os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# # llm = ChatGroq(model="llama3-8b-8192", groq_api_key = groq_api_key)


# # # Initialize Chroma vector store
# # import chromadb
# # client = chromadb.Client()
# # vector_store = Chroma(collection_name="domain_content", client=client)

# # #url = "https://www.alintaenergy.com.au/"
# # def store_content_in_vector_db(url):

# #     # Start scraping from the root domain
# #     scraped_data = crawl_domain(url)

# #     print(scraped_data)
# #     # Initialize Hugging face embeddings
# #     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# #     # Embed and store the content
# #     for page in scraped_data:
# #         title = page['title']
# #         body = " ".join(page['body'])  # Combine the body content into one string
# #         text = f"Title: {title}\nContent: {body}"
        
# #         # Generate embedding for the text (title + body)
# #         embedding = embeddings.embed(text)
        
# #         # Add the embedding to the Chroma vector store
# #         vector_store.add_texts([text], embeddings=[embedding], metadatas=[{'url': page['url']}])
        
# #     return vector_store
# #     print("Content embedded and stored in Chroma.")


# import chromadb
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# # Define ChromaDB path (Persistent storage)
# DB_PATH = "./chroma_db"
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# # Initialize ChromaDB
# chroma_client = chromadb.PersistentClient(path=DB_PATH)
# vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# def get_retriever():
#     """Return a retriever instance for querying the vector store."""
#     return vector_store.as_retriever()



# from bs4 import BeautifulSoup
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from sentence_transformers import SentenceTransformer
# os.environ['CURL_CA_BUNDLE'] = ''  # This disables SSL verification globally
# os.environ['SSL_VERIFY'] = '0'

# import ssl
# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.poolmanager import PoolManager
# import re

# class TLSAdapter(HTTPAdapter):
#     def init_poolmanager(self, *args, **kwargs):
#         ctx = ssl.create_default_context()
#         ctx.check_hostname = False
#         ctx.verify_mode = ssl.CERT_NONE
#         kwargs['ssl_context'] = ctx
#         return PoolManager(*args, **kwargs)

# session = requests.Session()
# session.mount('https://', TLSAdapter())

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# #embeddings = HuggingFaceEmbeddings(model_name="./local_model", model_kwargs={"device": "cpu"})

# #embeddings = SentenceTransformer("C:\\SelfLearning\\DS\\GenerativeAI\\embedding_local_model")
# from langchain.embeddings.base import Embeddings

# class CustomSentenceTransformerEmbeddings(Embeddings):
#     def __init__(self, model_name: str = "C:\\SelfLearning\\DS\\GenerativeAI\\embedding_local_model"):
#         self.model = SentenceTransformer(model_name)

#     def embed_documents(self, texts):
#         return self.model.encode(texts, convert_to_numpy=True).tolist()

#     def embed_query(self, text):
#         return self.model.encode(text, convert_to_numpy=True).tolist()

# # Initialize custom embeddings
# embeddings = CustomSentenceTransformerEmbeddings()


# persist_directory = "./db"
# # Initialize vector DB
# vector_db = Chroma(persist_directory=persist_directory,collection_name="website_docs", embedding_function=embeddings)

# def store_in_vector_db(url, html):
#     """Processes scraped content and stores in the vector database."""
#     print('store called')
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     soup = BeautifulSoup(html, "html.parser")
#     text = soup.get_text(separator=" ", strip=True)
#     text = clean_text(text)
    
#     if text:
#         chunks = text_splitter.split_text(text)
#         metadatas = [{"source": url} for _ in chunks]
#         vector_db.add_texts(texts=chunks, metadatas=metadatas)


# def clean_text(text):
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
#     text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
#     return text


from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
import numpy as np
from typing import List, Dict, Optional
import os
import re
import hashlib
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
import torch
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OptimizedSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, 
                 model_name: str ='sentence-transformers/all-MiniLM-L6-v2', #"C:\\SelfLearning\\DS\\GenerativeAI\\embedding_local_model",
                 batch_size: int = 32,
                 device: Optional[str] = None):
        """
        Initialize the embedding model with optimizations.
        
        Args:
            model_name: Path to the local model or name of the model
            batch_size: Number of texts to process at once
            device: Device to run the model on ('cuda' if available, else 'cpu')
        """
        self.batch_size = batch_size
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        
        # Enable half precision for CUDA devices
        if self.device == 'cuda':
            self.model.half()  # Convert to FP16 for faster processing
        
        # Cache for frequently accessed embeddings
        self.cache = {}
        self.cache_size = 10000  # Maximum number of cached embeddings

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text string."""
        return hashlib.md5(text.encode()).hexdigest()

    def _batch_encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts with progress bar."""
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device
            )
            embeddings.extend(batch_embeddings.tolist())
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        # Check cache first
        cached_embeddings = []
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                cached_embeddings.append((i, self.cache[cache_key]))
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # Generate new embeddings for uncached texts
        if texts_to_embed:
            new_embeddings = self._batch_encode(texts_to_embed)
            
            # Update cache
            for text, embedding in zip(texts_to_embed, new_embeddings):
                cache_key = self._get_cache_key(text)
                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache)))  # Remove oldest item
                self.cache[cache_key] = embedding

        # Combine cached and new embeddings in correct order
        all_embeddings = [None] * len(texts)
        for i, embedding in cached_embeddings:
            all_embeddings[i] = embedding
        for i, embedding in zip(indices_to_embed, new_embeddings):
            all_embeddings[i] = embedding

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        embedding = self.model.encode(text, convert_to_numpy=True, device=self.device).tolist()
        
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = embedding
            
        return embedding

class OptimizedVectorStore:
    def __init__(self, persist_directory: str = "./db"):
        """Initialize the optimized vector store."""
        self.persist_directory = persist_directory
        self.embeddings = OptimizedSentenceTransformerEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize Chroma with optimized settings
        self.vector_db = Chroma(
            persist_directory=persist_directory,
            collection_name="website_docs",
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Document deduplication cache
        self.content_hashes = set()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text with improved preprocessing."""
        # Remove HTML tags if any remain
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Normalize sentence boundaries
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        
        return text.strip()

    def get_content_hash(self, text: str) -> str:
        """Generate a hash for content deduplication."""
        # Normalize text for consistent hashing
        normalized_text = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized_text.encode()).hexdigest()

    def store_in_vector_db(self, url: str, html: str, batch_size: int = 100):
        """Store content in vector database with optimizations."""
        try:
            # Extract and clean text
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove unwanted elements
            for element in soup.select('script, style, meta, link'):
                element.decompose()
            
            text = soup.get_text(separator=" ", strip=True)
            text = self.clean_text(text)
            
            if not text:
                logging.warning(f"No valid text content found for URL: {url}")
                return
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Deduplicate chunks
            unique_chunks = []
            for chunk in chunks:
                chunk_hash = self.get_content_hash(chunk)
                if chunk_hash not in self.content_hashes:
                    self.content_hashes.add(chunk_hash)
                    unique_chunks.append(chunk)
            
            if not unique_chunks:
                logging.info(f"No unique content found for URL: {url}")
                return
            
            # Prepare metadata
            metadatas = [
                {
                    "source": url,
                    "chunk_index": i,
                    "total_chunks": len(unique_chunks)
                }
                for i in range(len(unique_chunks))
            ]
            
            # Store in batches
            for i in range(0, len(unique_chunks), batch_size):
                batch_chunks = unique_chunks[i:i + batch_size]
                batch_metadata = metadatas[i:i + batch_size]
                
                self.vector_db.add_texts(
                    texts=batch_chunks,
                    metadatas=batch_metadata
                )
            
            logging.info(f"Successfully stored {len(unique_chunks)} chunks from {url}")
            
        except Exception as e:
            logging.error(f"Error storing content from {url}: {e}")
            raise

    def similarity_search(self, query: str, k: int = 4) -> List[Dict]:
        """Perform optimized similarity search."""
        try:
            # Clean query
            query = self.clean_text(query)
            
            # Get results
            results = self.vector_db.similarity_search_with_relevance_scores(
                query,
                k=k * 2  # Fetch more results than needed for post-processing
            )
            
            # Post-process results
            processed_results = []
            seen_content = set()
            
            for doc, score in results:
                content_hash = self.get_content_hash(doc.page_content)
                if content_hash not in seen_content and score > 0.7:  # Adjust threshold as needed
                    seen_content.add(content_hash)
                    processed_results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'relevance_score': score
                    })
                    
                    if len(processed_results) >= k:
                        break
            
            return processed_results
            
        except Exception as e:
            logging.error(f"Error during similarity search: {e}")
            raise

# Usage example
if __name__ == "__main__":
    vector_store = OptimizedVectorStore()
    
    # Example usage for storing content
    # vector_store.store_in_vector_db(url="example.com", html="<html>...</html>")
    
    # Example search
    # results = vector_store.similarity_search("your query here")