import urllib.parse
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, JSONResponse
import requests, uvicorn, urllib
from helper_functions import generate_code_challenge, generate_code_verifier, generate_nonce, generate_state

from dotenv import load_dotenv
import os
import requests
from helper_functions import *
#Load env variables
load_dotenv()

#Okta Config
OKTA_DOMAIN = os.getenv("OKTA_DOMAIN")
OKTA_CLIENT_ID = os.getenv("OKTA_CLIENT_ID")
OKTA_AUTHORIZE_URI = f"{OKTA_DOMAIN}/v1/authorize"
OKTA_REDIRECT_URI = os.getenv("OKTA_REDIRECT_URI")
OKTA_TOKEN_URL = f"{OKTA_DOMAIN}/v1/token"
OKTA_USERINFO_URL = f"{OKTA_DOMAIN}/v1/userinfo"
OKTA_LOGOUT_URL = f"{OKTA_DOMAIN}/v1/logout"
OKTA_SCOPE = "openid profile email myaccount"

app = FastAPI()

# In memory storage for DEV, should be replaced with Redis or DB in prod
sessions = {}


# Login endpoint
@app.get("/login")
async def login():
    state = generate_state()
    nonce = generate_nonce()
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)

     # Store session info
    sessions[state] = {"code_verifier": code_verifier}

    okta_login_url = (
        f"{OKTA_DOMAIN}/v1/authorize?"
        f"client_id={OKTA_CLIENT_ID}&"
        f"code_challenge={code_challenge}&"
        f"code_challenge_method=S256&"
        f"nonce={nonce}&"
        #f"prompt=none&"
        f"redirect_uri={urllib.parse.quote(OKTA_REDIRECT_URI).replace("/", "%2F")}&"
        f"response_type=code&"
        #f"response_mode=okta_post_message&"
        f"state={state}&"
        f"scope={'openid profile email myaccount'}"
        )

    return RedirectResponse(okta_login_url)
    
# Callback endpoint
@app.get("/login/callback")
async def callback(request: Request):
    try:
        query_params = request.query_params
        code = query_params.get("code")
        state = query_params.get("state")

        if not code or not state or state not in sessions:
            return JSONResponse({"error": "Invalid state or code"}, status_code=400)

        # Get stored code_verifier
        code_verifier = sessions[state]["code_verifier"]

        # Exchange authorization code for tokens
        token_data = {
            "grant_type": "authorization_code",
            "client_id": OKTA_CLIENT_ID,
            "redirect_uri": OKTA_REDIRECT_URI,
            "code": code,
            "code_verifier": code_verifier
        }

        headers={
                    "Content-Type": "application/x-www-form-urlencoded"
        }

        token_response = requests.post(OKTA_TOKEN_URL, data=token_data, headers=headers)

        if token_response.status_code != 200:
            print("TOKEN EXCHANGE FAILED:", token_response.json())  # Log error response
            return JSONResponse({"error": "Token exchange failed"}, status_code=400)

        tokens = token_response.json()
        sessions[state]["tokens"] = tokens

        # Redirect back to Streamlit frontend with state
        return RedirectResponse(f"http://localhost:8501?state={state}")
    except Exception as e:
        print(e)
        return JSONResponse({"error": "Token exchange failed"})


# Get token endpoint
@app.get("/get_token")
async def get_token(state: str):
    if state not in sessions or "tokens" not in sessions[state]:
        return JSONResponse({"error": "Session expired or invalid state"}, status_code=400)

    return JSONResponse(sessions[state]["tokens"])


# Run server
if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8200)