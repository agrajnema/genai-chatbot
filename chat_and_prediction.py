import sys, os, httpx, ssl, time
from vectorstore import OptimizedVectorStore
import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional, Dict
import json
from datetime import datetime
import requests
import re
from dotenv import load_dotenv
from prompts import *
#from helper_functions import extract_pdf_content
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import chromadb
from langchain.schema import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
import chromadb
import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from helper_functions import extract_text_from_pdf

#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#embeddings = HuggingFaceEmbeddings(model_name="./local_model", model_kwargs={"device": "cpu"})
#embeddings = SentenceTransformer("C:\\SelfLearning\\DS\\GenerativeAI\\embedding_local_model")
#from llama_cpp import Llama
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain_community.llms import Ollama
from langchain.schema import AIMessage


import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
#from chat_usage_prediction import predict_usage
from langchain.embeddings.base import Embeddings

class CustomSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "C:\\SelfLearning\\DS\\GenerativeAI\\embedding_local_model"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()

# Initialize custom embeddings
embeddings = CustomSentenceTransformerEmbeddings()


class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = ctx
        return PoolManager(*args, **kwargs)

session = requests.Session()
session.mount('https://', TLSAdapter())


#Current time in unix timestamp
now = datetime.now()
one_year_ago = now - timedelta(days=365)  # Or timedelta(years=1) for slightly better accuracy
current_timestamp = int(time.mktime(now.timetuple()))
one_year_ago_timestamp = int(time.mktime(one_year_ago.timetuple()))


@st.cache_data
def get_default_session_state():
    return {
        "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        "chat_history": [],
        "api_cache": {},
        "is_logged_in": False,
        "last_query": ""
    }

def init_session_state():
    if "session_data" not in st.session_state:
        st.session_state.session_data = get_default_session_state()

    # Map stored session data to Streamlit session_state
    for key, value in st.session_state.session_data.items():
        st.session_state[key] = value

# Pydantic model for API decision with parameter handling
class APIParameters(BaseModel):
    query_params: Dict[str, str] = Field(description="Query parameters needed")
    route_params: Dict[str, str] = Field(description="Route parameters needed")
    source: str = Field(description="Source of parameter (user input, context, or session)")

class APIDecision(BaseModel):
    needs_api_call: bool = Field(description="Whether an API call is needed")
    use_vector_db: bool = Field(default=False, description="Whether to use vector DB instead of API")
    api_endpoint: Optional[str] = Field(default=None, description="The API endpoint to call")
    parameters: Optional[APIParameters] = Field(default=None, description="Required parameters for the API call")
    reasoning: str = Field(description="Reasoning behind the decision")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OCP_APIM_SUBSCRIPTION_KEY = os.getenv("OCP-APIM-SUBSCRIPTION-KEY")
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

vector_db = OptimizedVectorStore()

def normalize_field_name(field: str) -> str:
    """Normalize field names by removing special characters and converting to lowercase"""
    return re.sub(r'[^a-z0-9]', '', field.lower())

def get_field_variations() -> Dict[str, List[str]]:
    """Define common variations of field names"""
    return {
        'phone': ['mobile', 'phone', 'mobilenumber', 'phonenumber', 'contact', 'contactnumber'],
        'account': ['account', 'accountid', 'accountnumber', 'accountno'],
        'name': ['name', 'fullname', 'firstname', 'lastname'],
        'email': ['email', 'emailaddress', 'mail'],
        'address': ['address', 'location', 'residence'],
        'balance': ['balance', 'amount', 'accountbalance'],
        'status': ['status', 'state', 'condition'],
        'bill': ['bill', 'invoice']
    }

def extract_relevant_info(query: str, api_response: dict) -> str:
    """Extract relevant information from API response based on user query"""
    try:
        #Normalize the query
        normalized_query = normalize_field_name(query)
        
        # Get field variations
        field_variations = get_field_variations()
        
        # Flatten the API response for easier searching
        if isinstance(api_response, list):  # Handle cases where API response is a list
            flattened_response = {f"item_{i}": item for i, item in enumerate(api_response)}
        else:
            flattened_response = flatten_dict(api_response)

        if not isinstance(flattened_response, dict):
            raise ValueError("Flattened response is not a dictionary")
        
        # Find matching fields
        matched_fields = {}
        for field_key, variations in field_variations.items():
            # Check if any variation appears in the query
            if any(var in normalized_query for var in variations):
                # Look for matching response fields
                for resp_key, value in flattened_response.items():
                    normalized_resp_key = normalize_field_name(resp_key)
                    if any(var in normalized_resp_key for var in variations):
                        matched_fields[resp_key] = value

        print(f"matched fields: {matched_fields}")
        if matched_fields:
            # Format the response
            #llm = ChatGroq(temperature=0, groq_api_key="your-groq-api-key")
            response = llm.invoke(
                field_mapping_prompt.format(
                    query=query,
                    api_data=json.dumps(matched_fields)
                )
            )
            if isinstance(response, AIMessage):
                response_text = response.content.strip()
            else:
                response_text = response.strip() 
            return response_text
        
        # response = llm.invoke(
        #         field_mapping_prompt.format(
        #             query=query,
        #             api_data=json.dumps(api_response)
        #         )
        #     )
        # return response.content

        print('not found')
        return "I couldn't find the specific information you're looking for in the available data."

    except Exception as e:
        st.error(f"Error extracting information: {str(e)}")
        return "I encountered an error while processing your request."

def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flatten a nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# API configurations
API_ENDPOINTS = {
     "profile": {"url": "https://alinta-myaccount-api-dev.azure-api.net/Domain-CustomerAcnt/customers/{customernumber}/profile", "route_params":["customernumber"], "query_params": [], "required_params": ["customernumber"]},
    "customer_information": {"url": "https://alinta-myaccount-api-dev.azure-api.net/Domain-CustomerAcnt/customers/{customernumber}", "route_params":["customernumber"], "query_params": [], "required_params": ["customernumber"]},
    "customer_account": {"url": "https://alinta-myaccount-api-dev.azure-api.net/Domain-CustomerAcnt/accounts/{accountnumber}", "route_params":["accountnumber"], "query_params": [], "required_params": ["accountnumber"]},
    "billing": {"url": "https://alinta-myaccount-api-dev.azure-api.net/Domain-Billing/bills/{accountnumber}", "route_params":["accountnumber"], "query_params": [], "required_params": ["accountnumber"]},
    "bill_download": {"url": "https://alinta-myaccount-api-dev.azure-api.net/Domain-billing/bill/{accountnumber}/invoice/{invoiceid}",  "route_params":["accountnumber", "invoiceid"], "query_params": [], "required_params": ["accountnumber", "invoiceid"]},

}

def extract_parameters_from_query(query: str, endpoint_config: dict) -> dict:
    """Extract parameters from the query using LLM"""
    parser = JsonOutputParser(pydantic_object=APIParameters)
    chain = parameter_extraction_prompt | llm | parser
    
    # Get chat history context
    context = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in st.session_state.chat_history[-5:]  # Last 5 messages for context
    ])
    
    result = chain.invoke({
        "query": query,
        "context": context,
        "required_params": json.dumps(endpoint_config["required_params"])
    })
    
    return result

def normalize_parameter_name(param_name: str) -> str:
    """Normalize parameter names (e.g., 'user id' -> 'user_id')"""
    # Remove special characters and spaces
    normalized = re.sub(r'[^a-zA-Z0-9]', '_', param_name.lower())
    # Remove consecutive underscores
    normalized = re.sub(r'_+', '_', normalized)
    # Remove trailing underscores
    normalized = normalized.strip('_')
    return normalized

def build_api_url(endpoint_config: dict, route_params: dict) -> str:
    """Build API URL with route parameters"""
    url = endpoint_config["url"]
    for param, value in route_params.items():
        url = url.replace(f"{{{param}}}", str(value))
    return url



def call_api(endpoint: str, route_params: dict, query_params: dict) -> Optional[dict]:
    """Make API call with route and query parameters"""
    endpoint_config = API_ENDPOINTS[endpoint]

    try:

        url = build_api_url(endpoint_config, route_params)
        print("inside call api")
        print(url)
        headers = {
            "Authorization": f"Bearer {st.session_state.tokens["access_token"]}",
            "OCP-APIM-SUBSCRIPTION-KEY": OCP_APIM_SUBSCRIPTION_KEY
        }
        #print(st.session_state.tokens["access_token"])
        if endpoint == "billing":
            url = url + f"?fromDateTime={one_year_ago_timestamp}&toDateTime={current_timestamp}&page=0&pageSize=1"
            print(f"##Billing URL: {url}")

        response = requests.request(
            method="get",
            url=url,
            params=query_params,
            headers=headers,
            verify=False
        )

        print(f"####Response from api: {response}")
        response.raise_for_status()
        data = response.json()
        print(data)

        if(endpoint == "billing"):
            if data:    
                # call api for bill_download and add all the parameters
                #returned_data = json.loads(data)
                print('returned data')
                print(data)
                if isinstance(data, list) and data:
                    vector_db.store_api_data(url, data)
                    record = data[0]
                    print(record)
                    acc_num = record.get("accountNumber")
                    invoice_id = record.get("invoiceId")
                    print(acc_num)
                    print(invoice_id)
                    if(invoice_id):
                        call_api("bill_download", {"accountnumber": acc_num, "invoiceid": invoice_id}, {})

        elif(endpoint == "bill_download"):
            if data:
                print('i am in bill download')
                data = extract_text_from_pdf(data["invoiceUrl"])
                vector_db.store_pdf_data(url, data)
        else:
            vector_db.store_api_data(url,data)
        return data
    except Exception as e:
        print(f"API call failed: {str(e)}")
        return None


def validate_parameters(required_params: List[str], available_params: dict) -> tuple[bool, List[str]]:
    """Validate if all required parameters are available"""
    missing_params = []
    for param in required_params:
        normalized_param = normalize_parameter_name(param)
        if normalized_param not in available_params:
            missing_params.append(param)
    return len(missing_params) == 0, missing_params


def determine_intent(query: str, last_intent: None):
    """Use LLM to classify query intent with context awareness."""
    print('inside determine_intent')
    response = llm.invoke(intent_prompt.format(query=query, last_intent=last_intent), stop=["\n"])
    print('intent response')
    print(response)
    #return response.strip().lower()
    if isinstance(response, AIMessage):
        content = response.content.strip()
    else:
        content = response.strip() 
    return content


def query_vector_db(query: str, vectorstore: Chroma) -> str:
    """Query the vector DB for documentation and help content"""
    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=st.session_state.memory
        )
        
        response = qa_chain.invoke({
            "question": query,
            "chat_history": st.session_state.chat_history
        })
        
        return response["answer"]
    
    except Exception as e:
        st.error(f"Error querying vector DB: {str(e)}")
        return "I encountered an error while searching the documentation."

def extract_api_info(query: str, context: str) -> Tuple[str, Dict[str, Dict[str, str]], List[str]]:
    """
    Determines which API endpoint to use and extracts known and missing parameters.

    Returns:
        - endpoint (str): The API endpoint.
        - known_params (dict): Dictionary with 'route_params' and 'query_params'.
        - missing_params (list): List of missing parameters.
    """
    print(f"extract_api_info: {query}")
    print(f"extract_api_info: {context}")

    try:
        # Retrieve chat history for additional context
        context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in st.session_state.chat_history[-4:]  # Last 4 messages for better context
            if msg['content'] is not None
        ])

        # Invoke LLM with prompt
        response = llm.invoke(extract_api_info_prompt.format(query=query, context=context))
        if isinstance(response, AIMessage):
            response_text = response.content.strip()
        else:
            response_text = response.strip() 
        
        print("########## Raw Response from LLM ##########")
        print(response_text)

        # Extract JSON from response using regex
        match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
        if match:
            response_text = match.group(1).strip()

        # Parse JSON response
        parsed_data = json.loads(response_text)

        # Extract API details
        endpoint = parsed_data.get("endpoint", "none")
        known_params = parsed_data.get("known_params", {"route_params": {}, "query_params": {}})
        missing_params = parsed_data.get("missing_params", [])

        # Ensure proper format
        if isinstance(missing_params, str):
            missing_params = [] if missing_params.lower() == "none" else [missing_params]

        print("### Extracted API Info ###")
        print(f"Endpoint: {endpoint}")
        print(f"Known Params: {known_params}")
        print(f"Missing Params: {missing_params}")

        return endpoint, known_params, missing_params

    except json.JSONDecodeError:
        print("Error: Failed to parse JSON. Raw response:", response_text)
    except Exception as e:
        print(f"Error occurred: {str(e)}")

    return None, None, None


def process_query(user_query: str) -> str:
    """Main query processing function"""
    try:
        # Step 1: Determine if this is a vector DB query not the pdf vector store, but general vector store for website data
        
        #if (user_query == "predict my usage"):
            #return predict_usage()
        
        is_logged_in = False
        if 'session_id' in st.session_state:
            is_logged_in = True
        
        response = vector_db.similarity_search(user_query, is_logged_in=is_logged_in)
        return response
    except Exception as e:
        st.error(f"Error: {str(e)}")
    

def format_api_response(query: str, api_response: dict) -> str:
    """Format API response in natural language"""
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", "Format the API response to answer the user's question naturally."),
        ("human", f"Query: {query}\nAPI Response: {json.dumps(api_response)}")
    ])
    
    response = llm.invoke(response_prompt)
    return response

# Initialize session state
# if 'memory' not in st.session_state:
#     st.session_state.memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True
#     )
# if 'api_response' not in st.session_state:
#     st.session_state.api_response = example_api_response  # For testing

# Streamlit UI
st.title("💬 Alinta Assistant")

# Initialize session state ONCE
if "initialized" not in st.session_state:
    init_session_state()
    st.session_state.initialized = True

# Chat interface
user_input = st.chat_input("You:", key="user_input")

if user_input:
    try:
        print(user_input)
        response = process_query(user_input)
        print('###########response')
        print(response)
        
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response['content']})
    except Exception as e:
        print(e)
# Display chat history
# for message in st.session_state.chat_history:
#     if message["role"] == "user":
#         st.write(f"**You: {message['content']}**")
#     else:
#         st.write(f"""<div style='color: blue;'>Assistant: {message['content']}</div>""", unsafe_allow_html=True)

for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <img src="https://cdn-icons-png.flaticon.com/512/847/847969.png" width="30" style="margin-right: 10px;">
            <div style="background-color: #E1E1E1; padding: 10px; border-radius: 10px; max-width: 70%;">
                <strong>You:</strong> {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: flex-start; margin-bottom: 10px;">
            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" width="30" style="margin-right: 10px;">
            <div style="background-color: lightblue; padding: 10px; border-radius: 10px; max-width: 90%;">
                <strong>AI:</strong> {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)



OKTA_DOMAIN = os.getenv("OKTA_DOMAIN")
OKTA_USERINFO_URL = f"{OKTA_DOMAIN}/v1/userinfo"
AUTH_API_SERVER = "http://localhost:8200"

# Login redirect - user authentication
if "tokens" not in st.session_state:
    if st.button("Login to MyAccount"):
        auth_url = f"{AUTH_API_SERVER}/login"
        st.write("Redirecting...")
        st.markdown(f'<meta http-equiv="refresh" content="0; url={auth_url}">', unsafe_allow_html=True)


# Handle callback
state = st.query_params.get("state")

if state and "tokens" not in st.session_state:
    # Fetch tokens from API
    response = requests.get(f"{AUTH_API_SERVER}/get_token", params={"state": state})

    if response.status_code == 200:
        st.session_state.tokens = response.json()
        #st.rerun()
    else:
        st.error("Authentication failed. Please try again.")

# Display user info
if "tokens" in st.session_state:
    
    if not st.session_state["is_logged_in"]:
        st.success("You are logged in!")
    access_token = st.session_state.tokens["access_token"]
    st.session_state["is_logged_in"] = True

    # Fetch user info
    user_info = requests.get(OKTA_USERINFO_URL,
                             headers={"Authorization": f"Bearer {access_token}"}, verify=False).json()

    if user_info:
        print('this is user info')
        #st.write(user_info)
        if "session_id" not in st.session_state:
            print('assigning')
            st.write(f"Hi, {user_info["firstName"]}! What would you like to know about your account with Alinta?")
            st.session_state.session_id = user_info["myAccountId"]
        
            api_response = call_api("profile", {"customernumber" : f"{st.session_state.session_id}"}, {})
            if api_response:
                #st.session_state.api_cache["profile"] = api_response
                account_ids = [account["id"] for account in api_response["customerAccounts"]]
                print(account_ids)
                if len(account_ids) > 0:
                    for account_id in account_ids:
                        call_api("billing", {"accountnumber" : f"{account_id}"}, {})

    if st.button("Logout"):
            st.session_state.clear()
            st.rerun()
else:
    st.session_state["logged_in"] = False
    st.session_state.pop("api_cache", None)

# if st.button("ClearCache"):
#     st.session_state.clear()
#     st.rerun()
