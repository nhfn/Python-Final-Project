import streamlit as st
import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from rag import initialize_vector_db, retrieve_invoice_context
from tools import verify_invoice

os.environ["GOOGLE_API_KEY"] = "AIzaSyA-3WCqC05nF9D6KQujhveJxnmRta7D_c8"

# --- 1. Load Data ---
@st.cache_resource
def load_database_from_file():
    with open("invoices.json", "r") as f:
        invoices = json.load(f)
    
    formatted_data = []
    for inv in invoices:
        items_str = ", ".join([f"{i['name']} (${i['price']})" for i in inv['items']])
        text = (f"Invoice #{inv['invoice_id']}: Vendor: {inv['vendor']}. "
                f"Total: ${inv['total']}. Tax: ${inv['tax']}. Items: {items_str}.")
        formatted_data.append(text)
    return initialize_vector_db(formatted_data)

GLOBAL_VECTORSTORE = load_database_from_file()

# --- 2. Failsafe Tools ---
@tool
def retrieve_invoice_data(query: str) -> str:
    """Search the vector database for invoice details."""
    return retrieve_invoice_context(query, GLOBAL_VECTORSTORE)

@tool
def verify_invoice_tool(input_data: str) -> str:
    """Deterministic math verification. Fulfills Non-LLM tool requirement."""
    try:
        # 1. Clean the string (strips markdown if the AI added it)
        clean_json = re.sub(r'```json|```', '', input_data).strip()
        invoice_dict = json.loads(clean_json)
        
        # 2. FIX: If the LLM nested the data (e.g. {"INV002": {...}}), unwrap it
        if "items" not in invoice_dict and len(invoice_dict) == 1:
            first_key = list(invoice_dict.keys())[0]
            invoice_dict = invoice_dict[first_key]
            
        # 3. Run deterministic logic
        result = verify_invoice(invoice_dict)
        return f"Status: {result['status']}"
    except Exception as e:
        return f"Tool Error: {str(e)}. Please try sending the raw JSON data."
    

# --- 3. UI & Persistence ---
st.set_page_config(page_title="SNAP Auditor", page_icon="📊")
st.title("📊 Smart Invoice Analyzer")

DB_FILE = "chat_history.json"
if "messages" not in st.session_state:
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                st.session_state.messages = json.load(f)
        except: st.session_state.messages = []
    else: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# --- 4. Orchestration ---
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)
agent_tools = [retrieve_invoice_data, verify_invoice_tool]

agent = create_agent(
    model=llm,
    tools=agent_tools,
    system_prompt=(
        "You are a professional auditor. ALWAYS use retrieve_invoice_data first. "
        "To check math, pass the EXACT JSON details to verify_invoice_tool."
        "Always respond in plain text, as you are a chat bot, and cant output fancy outputs."
    )
)

# --- 5. Chat Input & Agent Execution ---
if user_prompt := st.chat_input("EX: Verify Best Buy INV002"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # Prepare history for the agent
                formatted = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                response = agent.invoke({"messages": formatted})
                
                output_text = ""
                
                # Case 1: Standard LangChain dict with 'output'
                if isinstance(response, dict) and "output" in response:
                    output_text = response["output"]
                # Case 2: Dict with 'messages' list
                elif isinstance(response, dict) and "messages" in response:
                    last_msg = response["messages"][-1]
                    output_text = getattr(last_msg, 'content', str(last_msg))
                # Case 3: Direct AIMessage or list of messages
                elif isinstance(response, list) and len(response) > 0:
                    last_item = response[-1]
                    output_text = getattr(last_item, 'content', str(last_item))
                # Case 4: Fallback
                else:
                    output_text = str(response)

                # Final cleanup: If the output_text is still a list (Flash quirk)
                if isinstance(output_text, list) and len(output_text) > 0:
                    if isinstance(output_text[0], dict) and 'text' in output_text[0]:
                        output_text = output_text[0]['text']
                    else:
                        output_text = str(output_text[0])

                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
                
                with open(DB_FILE, "w") as f:
                    json.dump(st.session_state.messages, f)

            except Exception as e:
                st.error(f"Error: {e}")
