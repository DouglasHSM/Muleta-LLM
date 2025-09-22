# ===============================================================
# app.py - FINAL SUBMISSION VERSION WITH BASE64 AUTH
# To run locally: python -m streamlit run app.py
# ===============================================================

import streamlit as st
import google.generativeai as genai
from google.cloud import bigquery
from google.oauth2 import service_account
import os
from dotenv import load_dotenv
import json
import pandas as pd
import plotly.express as px
import base64 # Importa a biblioteca para decodificação

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QueryMaster AI for BigQuery",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TO HIDE STREAMLIT'S DEFAULT MENU AND FOOTER ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# --- 1. SETUP AND AUTHENTICATION (BASE64 METHOD) ---

try:
    # Get secrets from Streamlit Secrets
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    gcp_key_base64 = st.secrets["GCP_KEY_BASE64"] # Pega o segredo codificado

    # --- DECODING THE BASE64 SECRET ---
    # 1. Get the Base64 string and convert it back to bytes
    gcp_key_bytes = gcp_key_base64.encode("utf-8")
    # 2. Decode the Base64 bytes to get the original JSON bytes
    key_bytes = base64.b64decode(gcp_key_bytes)
    # 3. Decode the original JSON bytes into a string
    gcp_key_content = key_bytes.decode("utf-8")
    
    # Configure Gemini API key
    genai.configure(api_key=google_api_key)
    
    # Load the decoded JSON key content into a dictionary
    gcp_key_json = json.loads(gcp_key_content)
    
    # Get the project_id from the key file's content
    project_id = gcp_key_json['project_id']
    
    # Pass both the credentials AND the project_id to the client
    credentials = service_account.Credentials.from_service_account_info(gcp_key_json)
    client_bq = bigquery.Client(credentials=credentials, project=project_id)
    
    st.session_state.auth_success = True
    print("Authentication successful using Base64 decoded key!")

except KeyError as e:
    st.error(f"Authentication Error: Secret '{e.args[0]}' not found. Please check your Streamlit Secrets. Did you name it GCP_KEY_BASE64?")
    st.stop()
except Exception as e:
    st.error(f"Authentication Error. The Base64 key might be corrupted. Details: {e}")
    st.stop()
    
# --- SYSTEM INSTRUCTION FOR THE LLM ---
SYSTEM_INSTRUCTION = """
You are 'QueryMaster', an AI Data Analyst specializing in the Google BigQuery 'TheLook' e-commerce dataset.
Your mission is to transform business questions into valid BigQuery Standard SQL queries.

You will use the following main schema:
CREATE TABLE `bigquery-public-data.thelook_ecommerce.order_items` (
  order_id STRING, user_id STRING, product_id STRING, sale_price NUMERIC, created_at TIMESTAMP
);
CREATE TABLE `bigquery-public-data.thelook_ecommerce.products` (
  id STRING, cost NUMERIC, category STRING, name STRING, brand STRING, department STRING
);
CREATE TABLE `bigquery-public-data.thelook_ecommerce.users` (
  id STRING, email STRING, first_name STRING, last_name STRING
);

Your response MUST be a valid JSON object.
The JSON must have the keys "action" and "content".
- For vague questions, use: {"action": "CLARIFY", "content": "Your clarification question here."}
- To generate SQL, use: {"action": "EXECUTE", "content": "Your SQL query here.", "display_format": "..."}

Possible values for "display_format": "currency_usd", "percentage", "number".

IMPORTANT: To group data by month and year from a TIMESTAMP column like 'created_at', use the function FORMAT_TIMESTAMP('%Y-%m', created_at). NEVER use the strftime function.
"""

# Configure the Gemini Model if authentication was successful
if 'auth_success' in st.session_state and st.session_state.auth_success:
    model = genai.GenerativeModel("gemini-2.5-pro", system_instruction=SYSTEM_INSTRUCTION)
    
    # --- 2. BACKEND LOGIC (HELPER FUNCTIONS) ---

    def clean_json_from_string(text):
        """Extracts a JSON string from within a Markdown code block."""
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1:
            return text[start_index:end_index+1]
        return text

    @st.cache_data
    def get_assistant_response(user_prompt, history_tuple):
        """
        This function encapsulates the backend logic. It calls the LLM and, if required, BigQuery.
        """
        history = [json.loads(item) for item in history_tuple]
        try:
            chat_session = model.start_chat(history=history)
            response = chat_session.send_message(user_prompt)
            cleaned_text = clean_json_from_string(response.text)
            
            response_json = json.loads(cleaned_text)
            action = response_json.get("action")

            if action == "EXECUTE":
                sql_query = response_json.get("content")
                display_format = response_json.get("display_format", "number")
                
                query_job = client_bq.query(sql_query)
                df_results = query_job.to_dataframe()
                
                return {
                    "action": "DATA", 
                    "content": df_results.to_dict('records'),
                    "query_used": sql_query, 
                    "display_format": display_format 
                }
            
            return response_json

        except json.JSONDecodeError:
            return {"action": "ERROR", "content": f"The model responded in an unexpected format: '{response.text}'"}
        except Exception as e:
            return {"action": "ERROR", "content": f"An error occurred: {e}"}

    def process_and_display_prompt(prompt):
        """Processes a prompt from the chat input or a button and displays the results in the UI."""
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data... Please wait."):
                history_for_cache = tuple(json.dumps(item) for item in st.session_state.history_for_api)
                response_data = get_assistant_response(prompt, history_for_cache)
                
                action = response_data.get("action")
                
                if action == "CLARIFY":
                    message_content = response_data["content"]
                    st.markdown(message_content)
                    st.session_state.messages.append({"role": "assistant", "content": message_content})

                elif action == "DATA":
                    # ... (resto da sua lógica de exibição, que já está perfeita) ...
                    data_content = response_data["content"]
                    display_format = response_data.get("display_format", "number")
                    # ...
                
                else: # Handle errors
                    message_content = f"An error occurred: {response_data.get('content')}"
                    st.error(message_content)
                    st.session_state.messages.append({"role": "assistant", "content": message_content})
        
        st.session_state.history_for_api.append({"role": "user", "parts": [prompt]})
        st.session_state.history_for_api.append({"role": "model", "parts": [json.dumps(response_data)]})

    # --- 3. STREAMLIT UI ---
    
    st.title("LLM Crutch 🤖: Your Data Analysis Assistant")
    # ... (o resto da sua UI, que já está perfeita) ...