# ===============================================================
# app.py - FINAL SUBMISSION VERSION (ENGLISH COMMENTS)
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

# Load environment variables from .env file for local development
load_dotenv()

# This authentication block is for local development (using ADC)
# It will be gracefully skipped in environments where ADC is not set up, like Streamlit Cloud
try:
    # Use st.secrets for Streamlit Cloud deployment, fallback to local ADC
    if "gcp_service_account" in st.secrets:
        # Load credentials from Streamlit Secrets
        gcp_key_json = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(gcp_key_json)
        client_bq = bigquery.Client(credentials=credentials, project=credentials.project_id)
        st.session_state.auth_method = "Streamlit Secrets"
    else:
        # Fallback to Application Default Credentials for local development
        client_bq = bigquery.Client()
        st.session_state.auth_method = "Local ADC"
    
    print(f"BigQuery Client initialized successfully via {st.session_state.auth_method}!")

except Exception as e:
    st.error(f"Error authenticating with Google Cloud. Details: {e}")
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

# Configure the Gemini Model
try:
    # Use st.secrets for Streamlit Cloud, fallback to os.getenv for local .env
    api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=SYSTEM_INSTRUCTION)
except Exception as e:
    st.error(f"Error configuring the Gemini API. Please check your GOOGLE_API_KEY secret/environment variable. Details: {e}")
    st.stop()


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
            
            # Execute the query using the initialized BigQuery client
            query_job = client_bq.query(sql_query)
            df_results = query_job.to_dataframe()
            
            # Return the results and metadata to the front-end
            return {
                "action": "DATA", 
                "content": df_results.to_dict('records'),
                "query_used": sql_query, 
                "display_format": display_format 
            }
        
        # If the action is CLARIFY or another, just return it
        return response_json

    except json.JSONDecodeError:
        return {"action": "ERROR", "content": f"The model responded in an unexpected format: '{response.text}'"}
    except Exception as e:
        return {"action": "ERROR", "content": f"An error occurred: {e}"}

def process_and_display_prompt(prompt):
    """Processes a prompt from the chat input or a button and displays the results in the UI."""
    # Append user message to the chat history for display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data... Please wait."):
            # Convert history to a hashable type for caching
            history_for_cache = tuple(json.dumps(item) for item in st.session_state.history_for_api)
            # Call the main backend logic function
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
                    # Case 1: The result is a single KPI
                    if df.shape[0] == 1 and df.shape[1] == 1:
                        st.markdown("#### Key Metric")
                        kpi_value = df.iloc[0, 0]
                        kpi_label = df.columns[0].replace("_", " ").title()
                        
                        # Format the KPI based on the LLM's hint
                        if display_format == "percentage": formatted_value = f"{kpi_value:,.2f}%"
                        elif display_format == "currency_usd": formatted_value = f"${kpi_value:,.2f}"
                        else: formatted_value = f"{kpi_value:,}"
                        st.metric(label=kpi_label, value=formatted_value)
                    # Case 2: The result is a table for charting
                    else:
                        col1, col2 = st.columns([1, 1.2]) # Make the chart column slightly larger
                        with col1:
                            st.markdown("#### Detailed Data")
                            st.dataframe(df)
                        with col2:
                            st.markdown("#### Chart")
                            try:
                                # Dynamically create a chart using Plotly
                                x_axis, y_axis = df.columns[0], df.columns[1]
                                fig = px.bar(df, x=x_axis, y=y_axis, title=f'{y_axis.replace("_", " ").title()} by {x_axis.replace("_", " ").title()}', template="seaborn")
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                st.warning("Could not generate a chart for this data.")
                
                with st.expander("View the generated SQL query"):
                    st.code(response_data["query_used"], language="sql")
                
                # Add a generic message to the display history
                message_content = "[Displaying data and charts]"
                st.session_state.messages.append({"role": "assistant", "content": message_content})
            
            else: # Handle errors
                message_content = f"An error occurred: {response_data.get('content')}"
                st.error(message_content)
                st.session_state.messages.append({"role": "assistant", "content": message_content})
    
    # Update the API history for the next turn
    st.session_state.history_for_api.append({"role": "user", "parts": [prompt]})
    st.session_state.history_for_api.append({"role": "model", "parts": [json.dumps(response_data)]})


# --- 3. STREAMLIT UI ---

st.title("LLM Crutch 🤖: Your Data Analysis Assistant")
st.caption("A project for the Kaggle BigQuery AI Hackathon by Douglas Menezes")

# --- SIDEBAR ---
st.sidebar.title("Analysis Suggestions 💡")
st.sidebar.markdown("Click a button to ask a sample question!")

if st.sidebar.button("📊 Monthly Profit (Chart)"): 
    process_and_display_prompt("Show me the monthly profit evolution for the year 2023.")

if st.sidebar.button("🏆 Top 5 Profitable Brands (Table)"): 
    process_and_display_prompt("What are the 5 most profitable brands?")
    
if st.sidebar.button("💰 Annual Growth (KPI %)") : 
    process_and_display_prompt("What was the percentage revenue growth between 2022 and 2023?")
    
if st.sidebar.button("🤯 Top Spenders Analysis (Complex JOIN)"): 
    process_and_display_prompt("List the top 3 users (with their emails) who spent the most on 'Jeans' products.")

st.sidebar.markdown("---")

st.sidebar.title("Controls")
if st.sidebar.button("🧹 Clear Conversation"):
    st.session_state.messages = []
    st.session_state.history_for_api = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    "**About:** 'LLM Crutch' is a conversational BI tool using Google's Gemini AI to translate natural language into SQL queries, executed on BigQuery."
)

# --- MAIN CHAT AREA ---
# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.history_for_api = []

# Display past messages from the history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get new user input
if prompt := st.chat_input("Ask your own question about the data..."):
    process_and_display_prompt(prompt)