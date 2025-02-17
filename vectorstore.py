from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
import numpy as np
from typing import List, Dict, Optional
import os
import re
import hashlib
import logging
from tqdm import tqdm
import torch
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import ssl, httpx
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize HTTP client with SSL verification disabled
client = httpx.Client(verify=False)
ssl._create_default_https_context = ssl._create_unverified_context


# Initialize LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    http_client=client
)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OptimizedSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', batch_size: int = 32, device: Optional[str] = None):
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name).to(self.device)

        if self.device == 'cuda':
            self.model.half()

    def _batch_encode(self, texts: List[str]) -> List[List[float]]:
        """Batch encode texts."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True, device=self.device).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._batch_encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._batch_encode([text])[0]
    
class OptimizedVectorStore:
    def __init__(self, persist_directory: str = "./db"):
        """Initialize the optimized vector store with multiple collections."""
        self.persist_directory = persist_directory
        self.embeddings = OptimizedSentenceTransformerEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize multiple collections
        self.collections = {
            "website_docs": Chroma(persist_directory=persist_directory, collection_name="website_docs", embedding_function=self.embeddings),
            "api_data": Chroma(persist_directory=None, collection_name="api_data", embedding_function=self.embeddings),
            "pdf_data": Chroma(persist_directory=None, collection_name="pdf_data", embedding_function=self.embeddings)
        }
    
    def determine_access_requirement(self, query: str) -> str:
        """Uses LLM to classify whether a query requires API/PDF access or scraped data is enough."""
        
        prompt = f"""
        Classify the following user query into one of the two categories:
        - "scraped_data" if it can be answered using publicly scraped content.
        - "restricted_data" if it requires information from API or PDFs (restricted access).
        Query: "{query}"
        
        Respond with only one word: "scraped_data" or "restricted_data".
        """
        print('inside determine')
        response = llm([HumanMessage(content=prompt)])
        print(response)
        intent = response.content.strip().lower()
        print(intent)
        return intent

    def similarity_search(self, query: str, k: int = 4, is_logged_in: bool = False, collections: Optional[List[str]] = None):
        """Decides query intent using LLM and performs search accordingly."""
        print('inside similarity')
        query = self.clean_text(query)
        print(query)
        # Step 1: Determine required access level
        access_required = self.determine_access_requirement(query) # type: ignore
        print('access')
        print(access_required)
        # Step 2: Enforce authentication if needed
        if access_required == "restricted_data" and not is_logged_in:
            return {"error": "🔒 You need to log in to access API/PDF data."}

    # Step 3: Select collections based on user status
        if access_required == "restricted_data" and is_logged_in:
            # Logged-in users can search in API & PDF collections
            collections = list(self.collections.keys())  # Search in all collections
        else:
            # Guests can only search in the scraped data collection
            collections = [self.collections["website_docs"]]

        results = []
        print(collections)
        print('total collections')
        print(self.collections)
        for collection_name in collections:
            print('collection name')
            #if collection_name in self.collections:
            #    print('hellllllo')
            try:
                print(collection_name)
                search_results = self.collections[collection_name].similarity_search_with_relevance_scores(query, k=k)
                print('search results')
                print(search_results)
                results.extend([(doc, score, collection_name) for doc, score in search_results])
            except Exception as e:
                logging.error(f"Error searching in {collection_name}: {e}")

        # Sort results by relevance score (higher is better)
        results = sorted(results, key=lambda x: x[1], reverse=True)

        # Deduplicate and return top-k
        seen_content = set()
        filtered_results = []
        
        for doc, score, collection in results:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                filtered_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'relevance_score': score,
                    'collection': collection
                })
                
            if len(filtered_results) >= k:
                break
        print(filtered_results)
        return filtered_results
    
    

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def fetch_or_retrieve_api_data(query: str, k: int = 5):
        """
        Check if API response is in the vector store before calling the API.
        If data exists in the vector store, return it. Otherwise, call the API, store, and return the data.

        Args:
            query (str): User query
            k (int): Number of top similar results to check

        Returns:
            dict: Retrieved API response or None
        """

    # Step 1: Search in vector store (API collection)
        results = vector_store.api_collection.similarity_search_with_relevance_scores(query, k=k)

        # Step 2: Check if a relevant result exists (adjust score threshold as needed)
        for doc, score in results:
            if score > 0.85:  # Adjust threshold based on quality of embeddings
                print("✅ API data found in vector store. Returning cached response.")
                return {"source": "vector_store", "data": doc.page_content}
        return None
            
    # def store_website_data(self, url: str, html: str):
    #     """Extract, clean, and store website data."""
    #     soup = BeautifulSoup(html, "html.parser")
    #     for element in soup.select('script, style, meta, link'):
    #         element.decompose()

    #     text = self.clean_text(soup.get_text(separator=" ", strip=True))
    #     chunks = self.text_splitter.split_text(text)

    #     metadata = [{"source": url, "chunk_index": i} for i in range(len(chunks))]
    #     print('inside scrape_page 5')
    #     self.collections["website_docs"].add_texts(texts=chunks, metadatas=metadata)
    #     print('inside scrape_page 6')
    #     logging.info(f"Stored {len(chunks)} chunks from website: {url}")

    def store_website_data(self, url: str, html: str):
        """Extract, clean, and store website data with improved error handling."""
        try:
            # Clean and chunk the text
            chunks = self.text_splitter.split_text(html)
            
            if not chunks:
                logging.warning(f"No chunks generated for {url}")
                return
                
            # Create metadata for each chunk
            metadata = [{"source": url, "chunk_index": i} for i in range(len(chunks))]
            
            # Store chunks in batches to prevent memory issues
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size]
                
                self.collections["website_docs"].add_texts(
                    texts=batch_chunks,
                    metadatas=batch_metadata
                )
                
                logging.info(f"Stored batch {i//batch_size + 1} of chunks for {url}")
                
            logging.info(f"Successfully stored {len(chunks)} chunks from {url}")
            
        except Exception as e:
            logging.error(f"Error storing data for {url}: {e}")
            raise

    def store_api_data(self, endpoint: str, response: Dict):
        """Store API response data in the vector store."""
        text = self.clean_text(str(response))  # Convert JSON to string
        chunks = self.text_splitter.split_text(text)

        metadata = [{"api_endpoint": endpoint, "chunk_index": i} for i in range(len(chunks))]
        self.collections["api_responses"].add_texts(texts=chunks, metadatas=metadata)

        logging.info(f"Stored {len(chunks)} chunks from API: {endpoint}")

    def store_pdf_data(self, pdf_name: str, pdf_text: str):
        """Store extracted PDF text in the vector store."""
        text = self.clean_text(pdf_text)
        chunks = self.text_splitter.split_text(text)

        metadata = [{"pdf_name": pdf_name, "chunk_index": i} for i in range(len(chunks))]
        self.collections["pdf_docs"].add_texts(texts=chunks, metadatas=metadata)

        logging.info(f"Stored {len(chunks)} chunks from PDF: {pdf_name}")

# Example Usage
if __name__ == "__main__":
    vector_store = OptimizedVectorStore()
