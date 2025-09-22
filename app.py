# ===============================================================
# To run: python -m streamlit run app.py
# ===============================================================

import streamlit as st
import google.generativeai as genai
from google.cloud import bigquery
import os
from dotenv import load_dotenv
import json
import pandas as pd
import plotly.express as px

# --- PAGE CONFIGURATION (FOR A PROFESSIONAL LOOK) ---
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

load_dotenv()

# Authenticate with BigQuery using Application Default Credentials (ADC)
try:
    client_bq = bigquery.Client()
    print("BigQuery Client initialized successfully via ADC!")
except Exception as e:
    st.error(f"Error authenticating with Google Cloud. Please ensure you have configured ADC. Details: {e}")
    st.stop()

# --- SYSTEM INSTRUCTION FOR THE LLM ---
SYSTEM_INSTRUCTION = """
You are 'QueryMaster', an AI Data Analyst specializing in the Google BigQuery 'TheLook' e-commerce dataset.
Your mission is to transform business questions about sales, products, and customers into valid BigQuery Standard SQL queries.

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
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-pro", system_instruction=SYSTEM_INSTRUCTION)
except Exception as e:
    st.error(f"Error configuring the Gemini API. Please check your GOOGLE_API_KEY. Details: {e}")
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
    """
    history = [json.loads(item) for item in history_tuple]
    try:
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_prompt)
        cleaned_text = clean_json_from_string(response.text)
        
        response_json = json.loads(cleaned_text)
        action = response_json.get("action")
        content = response_json.get("content")

        if action == "EXECUTE":
            sql_query = content
            display_format = response_json.get("display_format", "number")
            query_job = client_bq.query(sql_query)
            df_results = query_job.to_dataframe()
            results_json = df_results.to_json(orient='records')
            
            return {
                "action": "DATA", "content": json.loads(results_json),
                "query_used": sql_query, "display_format": display_format 
            }
        
        return response_json

    except json.JSONDecodeError:
        return {"action": "ERROR", "content": f"The model responded in an unexpected format: '{response.text}'"}
    except Exception as e:
        return {"action": "ERROR", "content": f"An error occurred: {e}"}

def process_and_display_prompt(prompt):
    """Processes a prompt from the chat input or a button and displays the results."""
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


# --- 3. STREAMLIT UI (THE "FRONT-END") ---

st.title("LLM Crutch | Your Data Analysis Assistant")
st.caption("A project for the Kaggle BigQuery AI Hackathon by Douglas Menezes")

# --- SIDEBAR ---
st.sidebar.title("Analysis Suggestions 💡")
st.sidebar.markdown(
    "Use these pre-made prompts to test the agent's capabilities. "
    "They are designed to showcase different levels of complexity."
)

st.sidebar.markdown("---")

# --- Routine Analysis Section ---
st.sidebar.subheader("Routine Business Analysis")
st.sidebar.caption("Typical questions a manager would ask to monitor performance.")

if st.sidebar.button("📊 Monthly Profit (Time-Series Chart)"): 
    process_and_display_prompt("Show me the monthly profit evolution for the year 2023.")

if st.sidebar.button("🏆 Top 5 Profitable Brands (Ranking)"): 
    process_and_display_prompt("What are the 5 most profitable brands?")
    
st.sidebar.markdown("---")

# --- Strategic Analysis Section ---
st.sidebar.subheader("Strategic & Complex Questions")
st.sidebar.caption("Questions that require the AI to perform complex calculations and reasoning.")
    
if st.sidebar.button("💰 Annual Growth (KPI %)") : 
    process_and_display_prompt("What was the percentage revenue growth between 2022 and 2023?")
    
if st.sidebar.button("🤯 Top Spenders Analysis (Complex JOIN)"): 
    process_and_display_prompt("List the top 3 users (with their emails) who spent the most on 'Jeans' products.")

st.sidebar.markdown("---")

# --- Utility Section ---
st.sidebar.title("Controls")
if st.sidebar.button("🧹 Clear Conversation"):
    st.session_state.messages = []
    st.session_state.history_for_api = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    "**About this project:** 'LLM Crutch' is a conversational BI tool that uses Google's Gemini AI to translate natural language questions into complex SQL queries, executing them on BigQuery to provide real-time data analysis and visualizations."
)

# MAIN CHAT AREA
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.history_for_api = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your own question about the data..."):
    process_and_display_prompt(prompt)