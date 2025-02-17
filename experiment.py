from langchain_chroma import Chroma
from vectorstore import OptimizedVectorStore

vectorStore = OptimizedVectorStore()
doc_count = vectorStore.collections["website_docs"]
docs = doc_count.get()
print(f"Count of documents: {doc_count}")
print(docs)

#vectorStore = OptimizedVectorStore()


# test_text= "This is a test document to check if vector storage works"
# vectorStore.store_website_data("https://test-url.com", test_text)

# print(vectorStore._collection.count())  # Should print a number > 0
# print(vectorStore._collection.peek(limit=5))  # Should show stored content