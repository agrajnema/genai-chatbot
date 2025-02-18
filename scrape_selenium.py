
##############################################


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from visited_url_sql_db import is_visited, mark_as_visited, init_db  # Assuming you have this module
from langchain.embeddings import SentenceTransformerEmbeddings


MAX_THREADS = 10  # Number of parallel threads
MODEL_NAME = "sentence-transformers/all-minilm-l6-v2"
CHROMA_PERSIST_DIR = "chroma_db"  # Directory for persistent ChromaDB
WEBSITE_COLLECTION_NAME = "website_docs"
API_COLLECTION_NAME = "api_data"
PDF_COLLECTION_NAME = "pdf_data"

# Initialize embeddings and collections

embeddings = SentenceTransformerEmbeddings(model_name=MODEL_NAME)

# Persistent Chroma for website data
website_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR, 
    embedding_function=embeddings, 
    collection_name=WEBSITE_COLLECTION_NAME
)

# In-memory Chroma for API and PDF data
api_db = Chroma(
    embedding_function=embeddings, 
    collection_name=API_COLLECTION_NAME
)  # No persist_directory for in-memory

pdf_db = Chroma(
    embedding_function=embeddings, 
    collection_name=PDF_COLLECTION_NAME
)  # No persist_directory for in-memory


# Initialize Sentence Transformer model
model = SentenceTransformer(MODEL_NAME)

# Chrome options (headless)
def get_chrome_options():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    return options

# Initialize ChromeDriver for each thread
def init_driver():
    options = get_chrome_options()
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

# Interact with dynamic content (accordions, buttons, dropdowns)
def interact_with_page(driver):
    try:
        # Improved interaction logic (more robust)
        for element_type in [By.XPATH, By.CSS_SELECTOR]:  # Try both XPATH and CSS
            elements = driver.find_elements(element_type, "button, select, .accordion") # Add accordion class
            for element in elements:
                try:
                    driver.execute_script("arguments[0].scrollIntoView();", element)
                    element.click()
                    time.sleep(1)
                except Exception as e:
                    print(f"Error interacting with {element_type}: {e}")
                    pass

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # Lazy loading
        time.sleep(2)
    except Exception as e:
        print(f"Error interacting with page: {e}")


def scrape_page(url):
    if is_visited(url):
        return None, set()

    print(f"Scraping: {url}")
    driver = init_driver()
    try:
        driver.get(url)
        time.sleep(2)
        interact_with_page(driver)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        page_text = soup.get_text(separator=" ", strip=True)

        links = set()
        base_domain = urlparse(url).netloc
        for a_tag in soup.find_all("a", href=True):
            link = urljoin(url, a_tag["href"])
            if urlparse(link).netloc == base_domain:
                links.add(link)

        mark_as_visited(url)
        if page_text: # Only add if text is present
            website_db.add_texts([page_text], ids=[url])  # Use LangChain Chroma's add_texts

        return page_text, links
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None, set()
    finally:
        driver.quit()


def scrape_domain(start_url, max_pages=50):
    """Multi-threaded scraping of all pages under a domain."""
    init_db()  # Ensure DB is initialized
    to_scrape = {start_url}
    scraped_count = 0

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        while to_scrape and scraped_count < max_pages:
            urls = list(to_scrape)[:MAX_THREADS]  # Process in batches
            to_scrape.difference_update(urls)  # Remove from queue

            # Multi-threaded execution
            results = executor.map(scrape_page, urls)

            for page_text, new_links in results:
                if page_text:
                    to_scrape.update(new_links - {start_url})  # Add new links
                    scraped_count += 1

    print("Scraping completed.")

def search_chroma(query, user_logged_in=False):
    website_results = website_db.similarity_search(query)  # LangChain Chroma's search
    all_results = website_results

    if user_logged_in:
        api_results = api_db.similarity_search(query)
        pdf_results = pdf_db.similarity_search(query)
        all_results.extend(api_results)
        all_results.extend(pdf_results)

    return all_results  # Returns list of Documents


if __name__ == "__main__":
    BASE_URL = "https://www.alintaenergy.com.au"
    init_db()  # Initialize the visited URLs database
    scrape_domain(BASE_URL, max_pages=1000)

    # Example search (replace with your actual query)
    query = "What are the Alinta Energy plans?"
    search_results = search_chroma(query, user_logged_in=True) # Set to True if user is logged in
    print(search_results)

    # To persist the vector database:
    client.persist()