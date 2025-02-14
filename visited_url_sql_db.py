import sqlite3

DB_NAME = "scraped_data.db"

def init_db():
    """Initialize SQLite database and create tables if not exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visited_urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE
        )
    """)
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Get a new SQLite connection for each thread."""
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def is_visited(url):
    """Check if a URL is already visited."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT 1 FROM visited_urls WHERE url = ?", (url,))
    result = cursor.fetchone()
    
    conn.close()
    return result is not None

def mark_as_visited(url):
    """Mark a URL as visited in SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("INSERT OR IGNORE INTO visited_urls (url) VALUES (?)", (url,))
    conn.commit()
    conn.close()


