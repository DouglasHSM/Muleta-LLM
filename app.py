# ===============================================================
# app.py - VERSÃO CEREBRAS (LLAMA 3.3) + BIGQUERY
# ===============================================================

import streamlit as st
from cerebras.cloud.sdk import Cerebras  # <--- NOVA IMPORTAÇÃO
from google.cloud import bigquery
from google.oauth2 import service_account
import os
from dotenv import load_dotenv
import json
import pandas as pd
import base64
from streamlit_echarts import st_echarts
import streamlit as st
from cerebras.cloud.sdk import Cerebras
from languages import TRANSLATIONS

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="BI Conversacional ",
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
    # Tenta pegar a chave do gerenciador de segredos do Streamlit (Prioridade)
    if "CEREBRAS_API_KEY" in st.secrets:
        cerebras_api_key = st.secrets["CEREBRAS_API_KEY"]
        # print("Chave carregada via secrets.toml") # Descomente para testar
    
    # Se não achar nos secrets, tenta ler do ambiente (Fallback para .env)
    else:
        load_dotenv() # Tenta carregar .env se existir
        cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
    
    # Validação Final
    if not cerebras_api_key:
        st.error("🚨 ERRO CRÍTICO: Chave API não encontrada!")
        st.warning("Para corrigir: Crie a pasta `.streamlit` e o arquivo `secrets.toml` com a chave `CEREBRAS_API_KEY`.")
        st.stop()
        
    # Inicializa o cliente Cerebras
    client_cerebras = Cerebras(api_key=cerebras_api_key)
    
    # --- Configuração do BigQuery (Mantida) ---
    if "GCP_KEY_BASE64" in st.secrets:
        gcp_key_base64 = st.secrets["GCP_KEY_BASE64"]
    elif os.path.exists('key_base64.txt'):
        with open('key_base64.txt', 'r') as f:
            gcp_key_base64 = f.read().strip()
    else:
        st.error("Chave do Google (BigQuery) não encontrada.")
        st.stop()

    # Decodifica BigQuery
    gcp_key_bytes = gcp_key_base64.encode("utf-8")
    key_bytes = base64.b64decode(gcp_key_bytes)
    gcp_key_content = key_bytes.decode("utf-8")
    gcp_key_json = json.loads(gcp_key_content)
    credentials = service_account.Credentials.from_service_account_info(gcp_key_json)
    client_bq = bigquery.Client(credentials=credentials, project=gcp_key_json['project_id'])
    
    st.session_state.auth_success = True

except Exception as e:
    st.error(f"Erro de Configuração: {e}")
    st.session_state.auth_success = False
    st.stop()
    
# --- SYSTEM INSTRUCTION (Otimizado para Llama) ---
SYSTEM_INSTRUCTION = """
Você é o 'QueryMaster', um Analista de Dados IA especialista no dataset de e-commerce 'TheLook'.
Sua missão é transformar perguntas de negócio em SQL (BigQuery Standard SQL).

SEUS MODOS DE OPERAÇÃO:
1. GERAÇÃO DE SQL: Se o usuário pedir novos dados, gere SQL (BigQuery Standard SQL).
2. ANÁLISE: Se o usuário perguntar sobre o gráfico já exibido (ex: "Por que caiu?", "Explique"), analise os dados do histórico e responda em texto (use action: "CLARIFY") RESPONDA COM UM ANALISTA BI SENIOR.

O schema é:
CREATE TABLE `bigquery-public-data.thelook_ecommerce.order_items` (order_id STRING, user_id STRING, product_id STRING, sale_price NUMERIC, created_at TIMESTAMP);
CREATE TABLE `bigquery-public-data.thelook_ecommerce.products` (id STRING, cost NUMERIC, category STRING, name STRING, brand STRING, department STRING);
CREATE TABLE `bigquery-public-data.thelook_ecommerce.users` (id STRING, email STRING, first_name STRING, last_name STRING, gender STRING);

RESPOSTA OBRIGATÓRIA (JSON):
{
  "action": "EXECUTE" | "CLARIFY",
  "content": "SQL Query" ou "Pergunta de esclarecimento",
  "display_format": "currency_brl" | "currency_usd" | "currency_eur" | "percentage" | "number" | "text",
  "chart_type": "bar" | "line" | "pie" | "scatter" | "table"
}

REGRAS DE SQL:
1. Agrupar datas: FORMAT_TIMESTAMP('%Y-%m', created_at).
2. Superlativos ("melhor", "top 1"): Use ORDER BY ... DESC LIMIT 1.
"""

# Continua apenas se a autenticação foi bem-sucedida
if 'auth_success' in st.session_state and st.session_state.auth_success:
    
    # --- 2. LÓGICA DO BACK-END (FUNÇÕES AUXILIARES) ---

    def clean_json_from_string(text):
        """Extrai uma string JSON de dentro de um bloco de código Markdown."""
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1:
            return text[start_index:end_index+1]
        return text

    def get_assistant_response(user_prompt, history_list):
        """
        Encapsula a lógica do back-end: chama a Cerebras e, se necessário, o BigQuery.
        """
        try:
            # 1. Montagem do Contexto (System + Histórico + User)
            messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
            
            # Adiciona histórico anterior
            messages.extend(history_list)
            
            # Adiciona pergunta atual
            messages.append({"role": "user", "content": user_prompt})

            # 2. Chamada API Cerebras
            completion = client_cerebras.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b",
                max_completion_tokens=1024,
                temperature=0.2, # Baixa temperatura para SQL preciso
                top_p=1,
                stream=False,
                response_format={"type": "json_object"} # Garante o JSON
            )

            # Processamento da Resposta
            response_text = completion.choices[0].message.content
            cleaned_text = clean_json_from_string(response_text)
            response_json = json.loads(cleaned_text)
            action = response_json.get("action")

            if action == "EXECUTE":
                sql_query = response_json.get("content")
                display_format = response_json.get("display_format", "number")
                chart_type = response_json.get("chart_type", "bar")
                
                # Executa no BigQuery
                query_job = client_bq.query(sql_query)
                df_results = query_job.to_dataframe()
                
                # Converte para string para evitar erros de serialização no histórico
                if not df_results.empty:
                    df_results = df_results.astype(str)
                
                return {
                    "action": "DATA", 
                    "content": df_results.to_dict('records'),
                    "query_used": sql_query, 
                    "display_format": display_format,
                    "chart_type": chart_type
                }
            
            return response_json

        except json.JSONDecodeError:
            return {"action": "ERROR", "content": f"Erro ao decodificar JSON do Llama: '{response_text}'"}
        except Exception as e:
            return {"action": "ERROR", "content": f"Ocorreu um erro técnico: {e}"}

    def process_and_display_prompt(prompt):
        """Processa um prompt (do chat ou botão) e exibe os resultados na UI."""
        # Adiciona mensagem do usuário na UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Llama 3.3 analisando dados..."):
                
                # Pega histórico formatado da sessão
                current_history = st.session_state.history_for_api
                response_data = get_assistant_response(prompt, current_history)
                
                action = response_data.get("action")
                
                if action == "CLARIFY":
                    message_content = response_data["content"]
                    st.markdown(message_content)
                    st.session_state.messages.append({"role": "assistant", "content": message_content})

                elif action == "DATA":
                    data_content = response_data["content"]
                    display_format = response_data.get("display_format", "number")

                    if not data_content:
                        st.warning("A consulta rodou, mas não retornou resultados.")
                        message_content = "Sem dados encontrados."
                    else:
                        df = pd.DataFrame(data_content)
                        # Tenta converter colunas numéricas de volta (pois viraram string antes)
                        for col in df.columns:
                            pd.to_numeric(df[col], errors='ignore')
                        
                        # --- DEFINE O SÍMBOLO DA MOEDA ---
                        currency_symbol = ""
                        if display_format == "currency_brl": currency_symbol = "R$ "
                        elif display_format == "currency_usd": currency_symbol = "$ "
                        elif display_format == "currency_eur": currency_symbol = "€ "

                        # --- KPI LÓGICA ---
                        if df.shape[0] == 1 and (df.shape[1] == 1 or df.shape[1] == 2):
                            st.markdown("#### Métrica Principal")
                            if df.shape[1] == 2:
                                kpi_label = df.columns[0]
                                kpi_value = df.iloc[0, 1]
                                st.markdown(f"**{kpi_label.replace('_', ' ').title()}:** {df.iloc[0, 0]}")
                            else:
                                kpi_label = df.columns[0].replace("_", " ").title()
                                kpi_value = df.iloc[0, 0]

                            # Tenta converter valor para float para formatação
                            try:
                                kpi_float = float(kpi_value)
                                if display_format == "percentage": 
                                    formatted_value = f"{kpi_float:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
                                elif display_format in ["currency_brl", "currency_usd", "currency_eur"]: 
                                    val_str = f"{kpi_float:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                                    formatted_value = f"{currency_symbol}{val_str}"
                                else: 
                                    formatted_value = f"{kpi_float:,}"
                            except:
                                formatted_value = str(kpi_value) # Fallback se não for número
                            
                            st.metric(label=kpi_label, value=formatted_value)
                        
                        # --- VISUALIZAÇÃO GRÁFICA ---
                        else:
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.markdown("#### Dados")
                                st.dataframe(df, use_container_width=True)
                            
                            with col2:
                                st.markdown("#### Visualização")
                                chart_type = response_data.get("chart_type", "bar")
                                x_col = df.columns[0]
                                y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                                
                                # Garante que y_col seja numérico para o gráfico
                                df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                                
                                colors = ['#FF4B4B', '#0068C9', '#83C9FF', '#FFABAB', '#29B09D', '#FF2B2B']

                                options = {
                                    "backgroundColor": "transparent",
                                    "tooltip": {
                                        "trigger": "axis", 
                                        "axisPointer": {"type": "shadow"},
                                        "valueFormatter": f"(value) => '{currency_symbol}' + value.toLocaleString()" 
                                    },
                                    "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
                                    "xAxis": {"type": "category", "data": df[x_col].astype(str).tolist(), "axisLabel": {"color": "#fff"}},
                                    "yAxis": {
                                        "type": "value", 
                                        "axisLabel": {
                                            "color": "#fff",
                                            "formatter": f"{currency_symbol}{{value}}" 
                                        }, 
                                        "splitLine": {"lineStyle": {"color": "#333"}}
                                    },
                                    "color": colors,
                                    "series": []
                                }

                                try:
                                    if chart_type == "line":
                                        st.info("📈 Tendência Temporal")
                                        options["series"] = [{
                                            "data": df[y_col].tolist(),
                                            "type": "line", "smooth": True,
                                            "areaStyle": {"opacity": 0.5}, "name": y_col
                                        }]
                                    elif chart_type == "pie":
                                        st.info("🍰 Distribuição")
                                        pie_data = [{"name": str(row[x_col]), "value": row[y_col]} for _, row in df.iterrows()]
                                        options["tooltip"] = {"trigger": "item", "valueFormatter": f"(value) => '{currency_symbol}' + value.toLocaleString()"}
                                        options["series"] = [{
                                            "name": x_col, "type": "pie", "radius": ["40%", "70%"],
                                            "avoidLabelOverlap": False,
                                            "itemStyle": {"borderRadius": 10, "borderColor": "#0e1117", "borderWidth": 2},
                                            "label": {"show": False, "position": "center"},
                                            "emphasis": {"label": {"show": True, "fontSize": 20, "fontWeight": "bold"}},
                                            "data": pie_data
                                        }]
                                        del options["xAxis"]; del options["yAxis"]; del options["grid"]
                                    elif chart_type == "scatter":
                                        st.info("💠 Dispersão")
                                        # --- CORREÇÃO AQUI ---
                                        # Força o eixo X a ser numérico, não texto
                                        options["xAxis"] = {
                                            "type": "value", 
                                            "axisLabel": {"color": "#fff", "formatter": f"{currency_symbol}{{value}}"},
                                            "splitLine": {"lineStyle": {"color": "#333"}}
                                        }
                                        # ---------------------
                                        scatter_data = df[[x_col, y_col]].values.tolist()
                                        options["series"] = [{"symbolSize": 20, "data": scatter_data, "type": "scatter", "name": "Dispersão"}]
                                    else: 
                                        options["series"] = [{"data": df[y_col].tolist(), "type": "bar", "showBackground": True, "backgroundStyle": {"color": "rgba(180, 180, 180, 0.2)"}, "name": y_col}]

                                    st_echarts(options=options, height="400px", theme="dark")
                                except Exception as e:
                                    st.warning(f"Erro visualização: {e}")
                    
                    with st.expander("Ver a consulta SQL gerada"):
                        st.code(response_data["query_used"], language="sql")
                    
                    message_content = "[Exibindo dados e gráficos]"
                    st.session_state.messages.append({"role": "assistant", "content": message_content})

                    if not df.empty:
                        # Converte os dados do gráfico para texto
                        dados_para_ia = df.head(30).to_csv(index=False)
                        
                        # Salva "escondido" no histórico para a IA ler depois
                        msg_sistema = {
                            "action": "DATA_RESULT",
                            "context": f"O gráfico foi gerado. Dados resultantes: {dados_para_ia}"
                        }
                        # Adiciona ao histórico que vai para a API (não aparece na tela)
                        st.session_state.history_for_api.append({
                            "role": "assistant", 
                            "content": json.dumps(msg_sistema)
                        })
                
                else: 
                    message_content = f"Ocorreu um erro: {response_data.get('content')}"
                    st.error(message_content)
                    st.session_state.messages.append({"role": "assistant", "content": message_content})
        
        # --- ATUALIZAÇÃO DO HISTÓRICO PARA API (FORMATO OPENAI/CEREBRAS) ---
        st.session_state.history_for_api.append({"role": "user", "content": prompt})
        # Salvamos o JSON completo da resposta como string no histórico para manter contexto da conversa
        st.session_state.history_for_api.append({"role": "assistant", "content": json.dumps(response_data)})

    # --- 3. INTERFACE DO STREAMLIT ---

    # ==========================================================
    # 🚑 CORREÇÃO: INICIALIZA O ESTADO ANTES DE TUDO
    # ==========================================================
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history_for_api" not in st.session_state:
        st.session_state.history_for_api = []
    # ==========================================================
    
    # --- SELETOR DE IDIOMA (NOVA FEATURE) ---
    lang_option = st.sidebar.selectbox("Language / Idioma", ["🇧🇷 Português", "🇺🇸 English"])
    
    if "Português" in lang_option:
        lang = "pt"
    else:
        lang = "en"
        
    # Carrega os textos do idioma selecionado
    t = TRANSLATIONS[lang]

    # --- TÍTULOS E CONTEXTO (USANDO AS VARIÁVEIS 't') ---
    st.title(t["title"])
    st.caption(t["caption"])
    
    st.markdown(t["welcome_title"])
    st.info(t["welcome_text"])

    with st.expander(t["expander_title"]):
        # Nota: O conteúdo do expander pode ser mantido fixo ou criar chaves para ele também
        if lang == "pt":
            st.markdown("""
            - **📦 Produtos:** Custos, categorias, marcas.
            - **💰 Vendas:** Receita, lucro, evolução.
            """)
        else:
            st.markdown("""
            - **📦 Products:** Costs, categories, brands.
            - **💰 Sales:** Revenue, profit, trends.
            """)
        
    st.divider()

    # --- BARRA LATERAL (BOTÕES TRADUZIDOS) ---
    st.sidebar.title(t["sidebar_header"])
    st.sidebar.markdown(t["sidebar_desc"])
    
    # Observe que usamos t['chave'] para o rótulo do botão E para o prompt enviado
    if st.sidebar.button(t["btn_line"]): 
        process_and_display_prompt(t["prompt_line"])
    
    if st.sidebar.button(t["btn_bar"]): 
        process_and_display_prompt(t["prompt_bar"])
    
    if st.sidebar.button(t["btn_pie"]): 
        process_and_display_prompt(t["prompt_pie"])

    if st.sidebar.button(t["btn_scatter"]): 
        process_and_display_prompt(t["prompt_scatter"])

    if st.sidebar.button(t["btn_table"]): 
        process_and_display_prompt(t["prompt_table"])
    
    st.sidebar.markdown("---")
    
    st.sidebar.title(t["controls"])
    if st.sidebar.button(t["btn_clear"]):
        st.session_state.messages = []
        st.session_state.history_for_api = []
        st.rerun()

    # ... (Renderização do histórico mantém igual) ...

    # Input do Chat Traduzido
    if prompt := st.chat_input(t["chat_placeholder"]):
        process_and_display_prompt(prompt)