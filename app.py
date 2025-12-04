# ===============================================================
# app.py - VERSÃO ATUALIZADA (PT-BR + PROMPT MELHORADO)
# Para rodar: python -m streamlit run app.py
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
import base64

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="BI Conversacional",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PARA ESCONDER O MENU E RODAPÉ ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# --- 1. AUTENTICAÇÃO E CONFIGURAÇÃO ---
try:
    # Procura pelos segredos do Streamlit Cloud (quando está no ar)
    if "GCP_KEY_BASE64" in st.secrets:
        print("Autenticando via Streamlit Secrets (Base64)...")
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        gcp_key_base64 = st.secrets["GCP_KEY_BASE64"]
        gcp_key_bytes = gcp_key_base64.encode("utf-8")
        key_bytes = base64.b64decode(gcp_key_bytes)
        gcp_key_content = key_bytes.decode("utf-8")
    
    # Fallback para o .env local (para quando você roda no seu PC)
    else:
        print("Autenticando via .env local...")
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        # Assume que você tem o arquivo key_base64.txt para desenvolvimento local
        with open('key_base64.txt', 'r') as f:
            gcp_key_base64 = f.read().strip()
        
        gcp_key_bytes = gcp_key_base64.encode("utf-8")
        key_bytes = base64.b64decode(gcp_key_bytes)
        gcp_key_content = key_bytes.decode("utf-8")

    # Configura os clientes da API
    genai.configure(api_key=google_api_key)
    gcp_key_json = json.loads(gcp_key_content)
    project_id = gcp_key_json['project_id']
    credentials = service_account.Credentials.from_service_account_info(gcp_key_json)
    client_bq = bigquery.Client(credentials=credentials, project=project_id)
    
    st.session_state.auth_success = True
    print("Autenticação realizada com sucesso!")

except Exception as e:
    st.error(f"Erro de Autenticação. Verifique seus segredos/credenciais. Detalhes: {e}")
    st.session_state.auth_success = False
    st.stop()
    
# --- SYSTEM INSTRUCTION ATUALIZADO (EM PT-BR E MAIS INTELIGENTE) ---
SYSTEM_INSTRUCTION = """
Você é o 'QueryMaster', um Analista de Dados IA especialista no dataset de e-commerce 'TheLook' do Google BigQuery.
Sua missão é transformar perguntas de negócio em português em queries SQL (dialeto BigQuery Standard SQL).

O schema principal que você usará é:
CREATE TABLE `bigquery-public-data.thelook_ecommerce.order_items` (
  order_id STRING, user_id STRING, product_id STRING, sale_price NUMERIC, created_at TIMESTAMP
);
CREATE TABLE `bigquery-public-data.thelook_ecommerce.products` (
  id STRING, cost NUMERIC, category STRING, name STRING, brand STRING, department STRING
);
CREATE TABLE `bigquery-public-data.thelook_ecommerce.users` (
  id STRING, email STRING, first_name STRING, last_name STRING
);

Suas respostas DEVEM ser um objeto JSON válido com as chaves "action", "content" e, para EXECUTE, "display_format".
- Para perguntas vagas: {"action": "CLARIFY", "content": "Sua pergunta de esclarecimento aqui."}
- Para gerar o SQL: {"action": "EXECUTE", "content": "Sua query SQL aqui.", "display_format": "..."}

Valores possíveis para "display_format": "currency_brl", "percentage", "number", "text".

REGRAS DE SQL IMPORTANTES:
1. Para agrupar por mês/ano (ex: "lucro mensal"), use FORMAT_TIMESTAMP('%Y-%m', created_at). NUNCA use strftime.
2. **REGRA DE SUPERLATIVO:** Se a pergunta pedir por um único item (ex: "qual o *melhor*", "o *mais* lucrativo", "o *top 1*" produto/marca/etc), sua query DEVE usar `ORDER BY ... DESC LIMIT 1`. Isso é um KPI, não um gráfico.
3. Para valores monetários, use o formato 'currency_brl'. Para percentagens, use 'percentage'.
"""

# Continua apenas se a autenticação foi bem-sucedida
if 'auth_success' in st.session_state and st.session_state.auth_success:
    model = genai.GenerativeModel("gemini-1.5-pro", system_instruction=SYSTEM_INSTRUCTION)
    
    # --- 2. LÓGICA DO BACK-END (FUNÇÕES AUXILIARES) ---

    def clean_json_from_string(text):
        """Extrai uma string JSON de dentro de um bloco de código Markdown."""
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1:
            return text[start_index:end_index+1]
        return text

    @st.cache_data
    def get_assistant_response(user_prompt, history_tuple):
        """Encapsula a lógica do back-end: chama o LLM e, se necessário, o BigQuery."""
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
            return {"action": "ERROR", "content": f"O modelo respondeu num formato inesperado: '{response.text}'"}
        except Exception as e:
            return {"action": "ERROR", "content": f"Ocorreu um erro: {e}"}

    def process_and_display_prompt(prompt):
        """Processa um prompt (do chat ou botão) e exibe os resultados na UI."""
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando os dados... Por favor, aguarde."):
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
                        st.warning("A consulta não retornou resultados.")
                    else:
                        df = pd.DataFrame(data_content)
                        # --- LÓGICA DE EXIBIÇÃO ATUALIZADA ---
                        # Se o resultado for 1 linha E 1 ou 2 colunas (ex: nome e valor), trate como KPI
                        if df.shape[0] == 1 and (df.shape[1] == 1 or df.shape[1] == 2):
                            st.markdown("#### Métrica Principal")
                            # Se tiver 2 colunas (ex: 'produto_nome', 'lucro_total'), usa uma como label e outra como valor
                            if df.shape[1] == 2:
                                kpi_label = df.columns[0]
                                kpi_value = df.iloc[0, 1]
                                st.markdown(f"**{kpi_label.replace('_', ' ').title()}:** {df.iloc[0, 0]}")
                            # Se tiver 1 coluna, usa o nome da coluna como label
                            else:
                                kpi_label = df.columns[0].replace("_", " ").title()
                                kpi_value = df.iloc[0, 0]

                            # Formatação (agora em R$)
                            if display_format == "percentage": formatted_value = f"{kpi_value:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
                            elif display_format == "currency_brl": formatted_value = f"R$ {kpi_value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                            else: formatted_value = f"{kpi_value:,}"
                            
                            st.metric(label=kpi_label, value=formatted_value)
                        
                        # Se for uma lista/tabela, mostra o gráfico e os dados
                        else:
                            col1, col2 = st.columns([1, 1.2])
                            with col1:
                                st.markdown("#### Dados Detalhados")
                                st.dataframe(df)
                            with col2:
                                st.markdown("#### Gráfico")
                                try:
                                    x_axis, y_axis = df.columns[0], df.columns[1]
                                    fig = px.bar(df, x=x_axis, y=y_axis, title=f'{y_axis.replace("_", " ").title()} por {x_axis.replace("_", " ").title()}', template="seaborn")
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception:
                                    st.warning("Não foi possível gerar um gráfico para estes dados.")
                    
                    with st.expander("Ver a consulta SQL gerada"):
                        st.code(response_data["query_used"], language="sql")
                    
                    message_content = "[Exibindo dados e gráficos]"
                    st.session_state.messages.append({"role": "assistant", "content": message_content})
                
                else: # Lida com erros
                    message_content = f"Ocorreu um erro: {response_data.get('content')}"
                    st.error(message_content)
                    st.session_state.messages.append({"role": "assistant", "content": message_content})
        
        st.session_state.history_for_api.append({"role": "user", "parts": [prompt]})
        st.session_state.history_for_api.append({"role": "model", "parts": [json.dumps(response_data)]})

    # --- 3. INTERFACE DO STREAMLIT (EM PT-BR) ---

    st.title(" </ Seu Assistente de Análise de Dados >")
    st.caption(f"Um projeto de Douglas Menezes (Construído para o Hackathon Kaggle BigQuery AI)")
    
    st.sidebar.title("Sugestões de Análise 💡")
    st.sidebar.markdown("Clique em um botão para fazer uma pergunta de teste!")
    if st.sidebar.button("📊 Lucro Mensal (Gráfico)"): process_and_display_prompt("Me mostre a evolução mensal do lucro no ano de 2023.")
    if st.sidebar.button("🏆 Top 5 Marcas Lucrativas (Tabela)"): process_and_display_prompt("Quais são as 5 marcas que mais nos deram lucro?")
    if st.sidebar.button("💰 Crescimento Anual (KPI %)") : process_and_display_prompt("Qual foi o crescimento percentual da receita entre 2022 e 2023?")
    if st.sidebar.button("🤯 Top Clientes (JOIN Complexo)"): process_and_display_prompt("Liste os 3 usuários (com seus e-mails) que mais gastaram em produtos da categoria 'Jeans'.")
    
    st.sidebar.markdown("---")
    
    st.sidebar.title("Controles")
    if st.sidebar.button("🧹 Limpar Conversa"):
        st.session_state.messages, st.session_state.history_for_api = [], []
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info("**Sobre:** 'Muleta para LLM' é uma ferramenta de BI conversacional que usa IA (Gemini) para traduzir linguagem natural em queries SQL, executadas no BigQuery.")

    if "messages" not in st.session_state:
        st.session_state.messages, st.session_state.history_for_api = [], []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("Faça sua própria pergunta sobre os dados..."):
        process_and_display_prompt(prompt)