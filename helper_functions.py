import base64, secrets, hashlib, uuid
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain, create_history_aware_retriever #retriever which knows about chat history
from langchain.prompts import PromptTemplate
import pdfplumber
import requests
from io import BytesIO
import tempfile

#PDF
from langchain_community.document_loaders import PyPDFLoader
import requests


# --- PKCE HELPERS ---
def generate_code_verifier():
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b'=').decode()

def generate_code_challenge(code_verifier):
    digest = hashlib.sha256(code_verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b'=').decode()

def generate_nonce():
    return secrets.token_urlsafe(16)

def generate_state():
    return secrets.token_urlsafe(16)


def get_session_id():
    """Generates or retrieves the session ID from session state."""
    if "session_id" not in st.session_state:
        # Generate a new UUID if it doesn't exist
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def get_session_history(sessionId:str) -> BaseChatMessageHistory:
    if sessionId not in st.session_state.store:
        st.session_state.store[sessionId] = ChatMessageHistory()
    return st.session_state.store[sessionId]



def retrieve_conversational_chain(llm, retriever):
    context_system_prompt = (
                "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood "
                "without the chat history. Do not answer the question, just rephrase it if needed, otherwise return as is."
            )

    context_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")

        ]
    )

    # create chat history aware retriever
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_question_prompt)
    
    #Question answer prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant for question-answer tasks. Use the retrieved context to answer the question. If you don't know the answer, say I don't know {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # create chain and pass llm and prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # create rag chain with history aware retriever
    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)


    # Memory-aware conversational chain
    conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
            )
    
    return conversational_rag_chain


def extract_text_from_pdf(pdf_url):
    try:
        # Download PDF from URL
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # Create a temporary file to save the PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(response.content)
            temp_path = temp_pdf.name
            
        # Process the PDF file
        bill_content = {}
        
        # Extract text using PyPDFLoader
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        
        # Process each page
        for i, page in enumerate(pages):
            bill_content[f"page_{i+1}"] = {
                'text': page.page_content,
                'metadata': page.metadata
            }
        
        # Extract tables using pdfplumber
        with pdfplumber.open(temp_path) as pdf:
            tables = []
            for i, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)
            bill_content['tables'] = tables
        
        print('bill_content')
        print(bill_content)
        return bill_content
        
    except requests.RequestException as e:
        print(f"Error downloading PDF: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None
    finally:
        # Clean up the temporary file
        import os
        try:
            os.unlink(temp_path)
        except:
            pass