from langchain_chroma import Chroma
from vectorstore import OptimizedVectorStore

# vectorStore = Chroma(persist_directory="./db")

# doc_count = vectorStore._collection.count()
# print(f"Count of documents: {doc_count}")

vectorStore = OptimizedVectorStore()


test_text= "This is a test document to check if vector storage works"
vectorStore.store_website_data("https://test-url.com", test_text)

print(vectorStore._collection.count())  # Should print a number > 0
print(vectorStore._collection.peek(limit=5))  # Should show stored content