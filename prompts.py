from langchain_core.prompts import ChatPromptTemplate


api_decision_prompt = ChatPromptTemplate.from_template("""
You are an API decision maker. Analyze the user query and decide how to handle it.
Based on the query: "{query}"
Login status: {is_logged_in}
Chat history: {chat_history}

Respond with a JSON object in exactly this format:
{{
    "needs_api_call": boolean,
    "use_vector_db": boolean,
    "api_endpoint": string or null,
    "parameters": {{
        "query_params": {{}},
        "route_params": {{}},
        "source": "user"
    }},
    "reasoning": string
}}
""")



# Create prompt templates
parameter_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract parameters from the user query and context. Consider:
    1. Route parameters that are part of the URL path
    2. Query parameters that are part of the URL query string
    3. Parameters that might be in the session state
    4. Parameters that need to be inferred from context
    5. When there are multiple records found, like if the user has multiple customerAccounts, present the accounts and ask which account are you enquiring for
    
    Map common variations of parameter names:
    - 'user id', 'user_id', 'userId' are the same
    - 'account number', 'account_id', 'accountId' are the same
    
    Format response as JSON with parameter values and sources."""),
    ("human", "Query: {query}\nContext: {context}\nRequired Parameters: {required_params}"),
])


# Field mapping prompt
field_mapping_prompt = ChatPromptTemplate.from_template("""
Given this user query: "{query}"
And this API response data: {api_data}

Extract the relevant information from the API response that answers the user's query.
Consider these field mappings:
- mobile, phone, mobile number, contact number all map to mobilePhone or phoneNumber
- account, account id, account number all map to accountId or accountNumber
- name, full name map to firstName + lastName or fullName
- email, email address map to emailAddress or email

Return only the relevant information in a natural sentence.
If no relevant information is found, say so clearly.
""")


intent_prompt = ChatPromptTemplate.from_template("""
    Analyze this query: {query}

    Return ONLY true or false based on these criteria:

    Return true if the query is about:
    - Company information, policies, or procedures
    - General service descriptions or offerings
    - How-to guides or documentation
    - Business hours or locations
    - General pricing or plans
    - Company contact information
    - FAQ or help topics
    - Product features or specifications
    - Employee benefits
    - What does Alinta own                                                
    - General information about Energy Industry

    Return false if the query is about:
    - Personal account details
    - Individual billing or payments
    - User's contact information
    - Account balance or usage
    - Personal service status
    - Individual's address or location
    - Customer-specific plans or services
    - Personal payment history
    - Individual's documents or statements

    Examples:
    "What are your business hours?" → true
    "How do I read my bill?" → true
    "What's my current balance?" → false
    "Can you explain your solar plans?" → true
    "What's my account number?" → false
    "What payment methods do you accept?" → true
    "When is my next bill due?" → false
    
    Answer with only true or false:
    """)

extract_api_info_prompt = ChatPromptTemplate.from_template("""
    Based on this user query: {query}
    Available context: {context}
                                                           
    FIRST determine if this is:
    A) A general information question that doesn't require customer-specific data
    B) A request for customer-specific information that requires an API call

    Respond **EXACTLY** in one of the following JSON formats (no extra text):

    **Format for A (general information):**
    ```json
    {{
    "endpoint": "none",
    "known_params": {{
        "route_params": {{}},
        "query_params": {{}}
    }},
    "missing_params": "none"
    }}

    If B (customer-specific), choose from these API endpoints:
    1. profile: For user profile information (requires: customernumber)
    2. customer_information: For general customer details (requires: customernumber)
    3. customer_account: For account-specific information (requires: accountnumber)
    4. billing: For bill and payment information (requires: accountnumber)
    5. bill_download: For downloading bill PDFs (requires: accountnumber, invoiceid)

    **Format for B (customer specific):**
    ```json
    {{
    "endpoint": "billing",  // Or other relevant endpoint
    "known_params": {{
        "route_params": {{"accountnumber": "[the account number if found]"}},
        "query_params": {{}}
    }},
    "missing_params": ["accountnumber"] // Or other missing parameters, or "none"
    }}
                                                           
    Examples of type A (general information):
    - Questions about wind farms
    - Questions about company history
    - Questions about general policies
    - Questions about energy industry

    Examples of type B (customer-specific):
    - Questions about bills
    - Questions about account balance
    - Questions about personal profile
    - Questions about payment history
    
    IMPORTANT: 
    1. If user provides a number in response to a request for an account number, treat it as accountnumber
    2. Look for numbers in both query and context that could be account numbers
    3. If the previous message asked for an account number and user responds with a number, use that as accountnumber
    4. Customer Number should already be there for a logged in user, try to derive it from the session state cache
    
    

    Examples:
    If context shows: "12345" after system asked for account number
    endpoint: billing
    known_params: {{
        "route_params": {{"accountnumber": "12345"}},
        "query_params": {{}}
    }}
    missing_params: none
""")


# extract_api_info_prompt = ChatPromptTemplate.from_template("""
#     Based on this user query: {query}
#     Available context: {context}

#     FIRST, determine if this is:
#     A) A general information question that doesn't require customer-specific data
#     B) A request for customer-specific information that requires an API call

#     If A (general information), respond with **ONLY** this JSON format:
#     ```json
#     {
#     "endpoint": "none",
#     "known_params": {
#         "route_params": {},
#         "query_params": {}
#     },
#     "missing_params": []
#     }
                                                           
#     If B (customer-specific), choose from these API endpoints:

#     profile: For user profile information (requires: customernumber)
#     customer_information: For general customer details (requires: customernumber)
#     customer_account: For account-specific information (requires: accountnumber)
#     billing: For bill and payment information (requires: accountnumber)
#     bill_download: For downloading bill PDFs (requires: accountnumber, invoiceid)
                                                           
#     If B (customer-specific), respond with **ONLY** this JSON format:
#     ```json
#     {
#         "endpoint": "[chosen endpoint]",
#         "known_params": {
#             "route_params": {
#             "[required_route_param]": "[value if found]"
#             },
#             "query_params": {}
#         },
#         "missing_params": ["list missing required params or empty list if none"]
#     }
#     """)
