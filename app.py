# ===============================================================
# app.py - FINAL SUBMISSION VERSION
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
import base64 # <-- Adicione esta importação

try:
    # Get secrets from Streamlit Secrets
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    gcp_key_base64 = st.secrets["GCP_KEY_BASE64"] # <-- Pega o novo segredo codificado

    # --- DECODIFICAÇÃO DO SEGREDO ---
    # 1. Pega a string Base64 e a converte de volta para bytes
    gcp_key_bytes = gcp_key_base64.encode("utf-8")
    # 2. Decodifica os bytes Base64 para os bytes do JSON original
    key_bytes = base64.b64decode(gcp_key_bytes)
    # 3. Decodifica os bytes do JSON para a string de texto original
    gcp_key_content = key_bytes.decode("utf-8")
    
    # Daqui em diante, o código é o mesmo de antes
    genai.configure(api_key=google_api_key)
    
    gcp_key_json = json.loads(gcp_key_content)
    project_id = gcp_key_json['project_id']
    credentials = service_account.Credentials.from_service_account_info(gcp_key_json)
    client_bq = bigquery.Client(credentials=credentials, project=project_id)
    
    st.session_state.auth_success = True
    print("Authentication successful using Base64 decoded key!")

except Exception as e:
    st.error(f"Authentication Error. Please check your Streamlit Secrets (Base64). Details: {e}")
    st.stop()

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


# --- 1. SETUP AND AUTHENTICATION ---
# This block handles authentication for both local development (using .env and ADC)
# and Streamlit Community Cloud deployment (using st.secrets).

# Load environment variables from .env file for local development
load_dotenv() 

try:
    # Check if Streamlit's secrets management is in use (for deployment)
    if "GCP_KEY" in st.secrets:
        print("Authenticating using Streamlit Secrets...")
        gcp_key_content = st.secrets["GCP_KEY"]
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    # Fallback to local .env file and ADC for local development
    else:
        print("Authenticating using local .env file and ADC...")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        # For local ADC, the client finds credentials automatically. We just need the project ID.
        # However, to keep the code consistent, we will use the key file method if it exists.
        # The most robust method that was confirmed to work is now the default.
        if os.path.exists('key.json'):
             with open('key.json') as f:
                gcp_key_content = f.read()
        else: # Fallback for pure ADC without a visible key file
            client_bq = bigquery.Client()
            st.session_state.auth_success = True


    # Configure Gemini API key from the loaded secret
    genai.configure(api_key=google_api_key)
    
    # Configure BigQuery credentials from the loaded JSON key content
    gcp_key_json = json.loads(gcp_key_content)
    project_id = gcp_key_json['project_id']
    credentials = service_account.Credentials.from_service_account_info(gcp_key_json)
    client_bq = bigquery.Client(credentials=credentials, project=project_id)
    
    st.session_state.auth_success = True
    print("Authentication successful!")

except Exception as e:
    st.error(f"Authentication Error. Please check your secrets/credentials setup. Details: {e}")
    st.session_state.auth_success = False
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
if st.session_state.auth_success:
    model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=SYSTEM_INSTRUCTION)
    
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
        The function is cached to save API calls for repeated questions.
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
                    data_content = response_data["content"]
                    display_format = response_data.get("display_format", "number")

                    if not data_content:
                        st.warning("The query returned no results.")
                    else:
                        df = pd.DataFrame(data_content)
                        if df.shape[0] == 1 and df.shape[1] == 1:
                            st.markdown("#### Key Metric")
                            kpi_value = df.iloc[0, 0]
                            kpi_label = df.columns[0].replace("_", " ").title()
                            
                            if display_format == "percentage": formatted_value = f"{kpi_value:,.2f}%"
                            elif display_format == "currency_usd": formatted_value = f"${kpi_value:,.2f}"
                            else: formatted_value = f"{kpi_value:,}"
                            st.metric(label=kpi_label, value=formatted_value)
                        else:
                            col1, col2 = st.columns([1, 1.2])
                            with col1:
                                st.markdown("#### Detailed Data")
                                st.dataframe(df)
                            with col2:
                                st.markdown("#### Chart")
                                try:
                                    x_axis, y_axis = df.columns[0], df.columns[1]
                                    fig = px.bar(df, x=x_axis, y=y_axis, title=f'{y_axis.replace("_", " ").title()} by {x_axis.replace("_", " ").title()}', template="seaborn")
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception:
                                    st.warning("Could not generate a chart for this data.")
                    
                    with st.expander("View the generated SQL query"):
                        st.code(response_data["query_used"], language="sql")
                    
                    message_content = "[Displaying data and charts]"
                    st.session_state.messages.append({"role": "assistant", "content": message_content})
                
                else: # Handle errors
                    message_content = f"An error occurred: {response_data.get('content')}"
                    st.error(message_content)
                    st.session_state.messages.append({"role": "assistant", "content": message_content})
        
        st.session_state.history_for_api.append({"role": "user", "parts": [prompt]})
        st.session_state.history_for_api.append({"role": "model", "parts": [json.dumps(response_data)]})

    # --- 3. STREAMLIT UI ---

    st.title("LLM Crutch 🤖: Your Data Analysis Assistant")
    st.caption("A project for the Kaggle BigQuery AI Hackathon by [Your Name Here]")
    
    st.sidebar.title("Analysis Suggestions 💡")
    st.sidebar.markdown("Click a button to ask a sample question!")
    if st.sidebar.button("📊 Monthly Profit (Chart)"): process_and_display_prompt("Show me the monthly profit evolution for the year 2023.")
    if st.sidebar.button("🏆 Top 5 Profitable Brands (Table)"): process_and_display_prompt("What are the 5 most profitable brands?")
    if st.sidebar.button("💰 Annual Growth (KPI %)") : process_and_display_prompt("What was the percentage revenue growth between 2022 and 2023?")
    if st.sidebar.button("🤯 Top Spenders Analysis (Complex JOIN)"): process_and_display_prompt("List the top 3 users (with their emails) who spent the most on 'Jeans' products.")
    
    st.sidebar.markdown("---")
    
    st.sidebar.title("Controls")
    if st.sidebar.button("🧹 Clear Conversation"):
        st.session_state.messages, st.session_state.history_for_api = [], []
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info("**About:** 'LLM Crutch' is a conversational BI tool using Google's Gemini AI to translate natural language into SQL queries, executed on BigQuery.")

    if "messages" not in st.session_state:
        st.session_state.messages, st.session_state.history_for_api = [], []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your own question about the data..."):
        process_and_display_prompt(prompt)