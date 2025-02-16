# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from bs4 import BeautifulSoup
# import time
# from urllib.parse import urljoin, urlparse
# from concurrent.futures import ThreadPoolExecutor
# from vectorstore import store_in_vector_db
# from visited_url_sql_db import is_visited, mark_as_visited, init_db

# MAX_THREADS = 10  # Number of parallel threads

# # Configure Chrome options
# def get_chrome_options():
#     options = Options()
#     options.add_argument("--headless")  # Run in headless mode
#     options.add_argument("--disable-gpu")
#     options.add_argument("--no-sandbox")
#     return options

# # Function to initialize ChromeDriver for each thread
# def init_driver():
#     """Initialize a new ChromeDriver instance for each thread."""
#     options = get_chrome_options()
#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
#     return driver

# def interact_with_page(driver):
#     """Click accordions, dropdowns, and buttons to reveal hidden content."""
#     try:
#         # Expand accordions
#         accordions = driver.find_elements(By.XPATH, "//button[contains(@class, 'accordion')]")
#         for accordion in accordions:
#             driver.execute_script("arguments[0].scrollIntoView();", accordion)
#             accordion.click()
#             time.sleep(1)  # Allow time for content to load

#         # Expand dropdowns
#         dropdowns = driver.find_elements(By.XPATH, "//select")
#         for dropdown in dropdowns:
#             driver.execute_script("arguments[0].scrollIntoView();", dropdown)
#             dropdown.click()
#             time.sleep(1)

#         # Click buttons that load additional content
#         buttons = driver.find_elements(By.XPATH, "//button")
#         for button in buttons:
#             try:
#                 driver.execute_script("arguments[0].scrollIntoView();", button)
#                 button.click()
#                 time.sleep(2)  # Wait for new content
#             except:
#                 pass  # Skip if not interactable

#         # Scroll to bottom for lazy-loaded content
#         driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(2)
#     except Exception as e:
#         print(f"Error interacting with page: {e}")

# def scrape_page(url):
#     """Scrape a webpage and extract text and internal links."""
#     if is_visited(url):
#         return None, set()  # Skip if already visited

#     print(f"Scraping: {url}")
#     driver = init_driver()
    
#     try:
#         driver.get(url)
#         time.sleep(2)  # Allow JavaScript to load
#         interact_with_page(driver)  # Reveal hidden content

#         # Extract content using BeautifulSoup
#         soup = BeautifulSoup(driver.page_source, "html.parser")

#         # Extract text content
#         page_text = soup.get_text(separator=" ", strip=True)

#         # Extract internal links
#         links = set()
#         base_domain = urlparse(url).netloc
#         for a_tag in soup.find_all("a", href=True):
#             link = urljoin(url, a_tag["href"])
#             if urlparse(link).netloc == base_domain:
#                 links.add(link)

#         mark_as_visited(url)  # Store URL in SQLite
#         store_in_vector_db(url, page_text)  # Store text in vector DB

#         return page_text, links
#     except Exception as e:
#         print(f"Error scraping {url}: {e}")
#         return None, set()
#     finally:
#         driver.quit()  # Ensure the driver is closed properly

# def scrape_domain(start_url, max_pages=50):
#     """Multi-threaded scraping of all pages under a domain."""
#     init_db()  # Ensure DB is initialized
#     to_scrape = {start_url}
#     scraped_count = 0

#     with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
#         while to_scrape and scraped_count < max_pages:
#             urls = list(to_scrape)[:MAX_THREADS]  # Process in batches
#             to_scrape.difference_update(urls)  # Remove from queue

#             # Multi-threaded execution
#             results = executor.map(scrape_page, urls)

#             for page_text, new_links in results:
#                 if page_text:
#                     to_scrape.update(new_links - {start_url})  # Add new links
#                     scraped_count += 1

#     print("Scraping completed.")

# if __name__ == "__main__":
#     BASE_URL = "https://www.alintaenergy.com.au"  # Change to your target website
#     scrape_domain(BASE_URL, max_pages=1000)  # Limit to 1000 pages



from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException, TimeoutException
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Tuple, Set, Optional
#from vectorstore import store_in_vector_db
from vectorstore import OptimizedVectorStore
from visited_url_sql_db import is_visited, mark_as_visited, init_db

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_THREADS = 5  # Reduced from 10 to prevent overwhelming the server
WAIT_TIME = 10  # Maximum wait time for elements
vector_store = OptimizedVectorStore()


def get_chrome_options() -> Options:
    """Configure Chrome options for headless browsing."""
    options = Options()
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--incognito")  # Optional: Starts Chrome in incognito mode
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    return options

def init_driver() -> webdriver.Chrome:
    """Initialize a new ChromeDriver instance with error handling."""
    try:
        options = get_chrome_options()
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.set_page_load_timeout(30)
        return driver
    except Exception as e:
        logging.error(f"Failed to initialize driver: {e}")
        raise

def safe_click(driver: webdriver.Chrome, element) -> bool:
    """Safely click an element with proper waits and error handling."""
    try:
        WebDriverWait(driver, WAIT_TIME).until(
            EC.element_to_be_clickable(element)
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        time.sleep(0.5)  # Short pause for smooth scrolling
        element.click()
        return True
    except (ElementClickInterceptedException, TimeoutException) as e:
        logging.warning(f"Could not click element: {e}")
        return False

def interact_with_page(driver: webdriver.Chrome):
    """Interact with dynamic page elements to reveal hidden content."""
    # List of common interactive elements
    interactive_elements = [
        ("//button[contains(@class, 'accordion')]", "accordion"),
        ("//div[contains(@class, 'collapse')]", "collapsible"),
        ("//select", "dropdown"),
        ("//button[contains(@class, 'load-more')]", "load more"),
        ("//button[not(@disabled)]", "button")
    ]

    wait = WebDriverWait(driver, WAIT_TIME)
    
    for xpath, element_type in interactive_elements:
        try:
            elements = driver.find_elements(By.XPATH, xpath)
            for element in elements:
                if safe_click(driver, element):
                    logging.info(f"Successfully interacted with {element_type}")
                    time.sleep(1)  # Wait for content to load
        except Exception as e:
            logging.warning(f"Error interacting with {element_type}: {e}")

    # Handle infinite scroll
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def scrape_page(url: str) -> Tuple[Optional[str], Set[str]]:
    """Scrape a webpage and extract text and internal links."""
    if is_visited(url):
        return None, set()

    driver = None
    try:
        logging.info(f"Scraping: {url}")
        driver = init_driver()
        driver.get(url)
        
        # Wait for the page to load
        wait = WebDriverWait(driver, WAIT_TIME)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        interact_with_page(driver)

        # Extract content using BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Remove unwanted elements
        for element in soup.select('script, style, meta, link'):
            element.decompose()

        # Extract text content
        page_text = ' '.join(soup.stripped_strings)

        # Extract internal links
        links = set()
        base_domain = urlparse(url).netloc
        for a_tag in soup.find_all("a", href=True):
            link = urljoin(url, a_tag["href"])
            if urlparse(link).netloc == base_domain and not link.endswith(('.pdf', '.jpg', '.png')):
                links.add(link)

        mark_as_visited(url)
        #store_in_vector_db(url, page_text)
        #vector_store.store_in_vector_db(url, html=page_text)

        if page_text:
            logging.info(f"Storing scraped content from {url} into vector DB. Length: {len(page_text)} characters")
            vector_store.store_in_vector_db(url, html=page_text)
        else:
            logging.warning(f"No text extracted from {url}, skipping storage")

        return page_text, links

    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return None, set()
    finally:
        if driver:
            driver.quit()

def scrape_domain(start_url: str, max_pages: int = 50):
    """Multi-threaded scraping of all pages under a domain with rate limiting."""
    init_db()
    to_scrape = {start_url}
    scraped = set()
    scraped_count = 0

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        while to_scrape and scraped_count < max_pages:
            current_batch = list(to_scrape)[:MAX_THREADS]
            to_scrape.difference_update(current_batch)

            # Execute batch
            future_to_url = {executor.submit(scrape_page, url): url for url in current_batch}
            for future in future_to_url:
                url = future_to_url[future]
                try:
                    page_text, new_links = future.result()
                    if page_text:
                        scraped.add(url)
                        to_scrape.update(new_links - scraped)
                        scraped_count += 1
                        logging.info(f"Successfully scraped {url}. Total: {scraped_count}/{max_pages}")
                except Exception as e:
                    logging.error(f"Error processing {url}: {e}")

            time.sleep(1)  # Rate limiting between batches

    logging.info(f"Scraping completed. Total pages scraped: {scraped_count}")

if __name__ == "__main__":
    BASE_URL = "https://www.alintaenergy.com.au"
    scrape_domain(BASE_URL, max_pages=10000)
