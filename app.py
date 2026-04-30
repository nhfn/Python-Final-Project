import streamlit as st
import os
import json
from rag import initialize_vector_db
from new_llm import get_invoice_agent

# Setup API Key using Streamlit Secrets for cloud deployment
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- 1. UI Setup ---
st.set_page_config(page_title="SNAP Auditor", page_icon="📊", layout="wide")

st.markdown("""
<style>
    /* Dark Mode Premium Theme */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .stChatInputContainer {
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 1rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    .stSidebar {
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(to right, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Smart Invoice Analyzer")

# --- 2. Load Data ---
@st.cache_resource
def load_database_from_file():
    if not os.path.exists("invoices.json"):
        return None
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



# --- 4. Sidebar & Persistence ---
DB_FILE = "chat_history.json"

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Manage your AI Auditor")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        st.rerun()

if "messages" not in st.session_state:
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                st.session_state.messages = json.load(f)
        except: st.session_state.messages = []
    else: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])

# --- 4. Orchestration ---
agent_executor = get_invoice_agent(GLOBAL_VECTORSTORE)
# --- 6. Chat Input & Execution ---
if user_prompt := st.chat_input("EX: Verify Best Buy INV002 or Find the highest invoice"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                chat_history = []
                for msg in st.session_state.messages:
                    chat_history.append({"role": msg["role"], "content": msg["content"]})
                
                response = agent_executor.invoke({
                    "messages": chat_history
                })
                
                output_text = ""
                if isinstance(response, dict) and "messages" in response and len(response["messages"]) > 0:
                    last_msg = response["messages"][-1]
                    raw_content = getattr(last_msg, "content", str(last_msg))
                    
                    if isinstance(raw_content, list):
                        # Extract text from list of dicts
                        text_parts = [part.get("text", "") for part in raw_content if isinstance(part, dict) and "text" in part]
                        output_text = "".join(text_parts)
                    elif isinstance(raw_content, str):
                        output_text = raw_content
                    else:
                        output_text = str(raw_content)
                else:
                    output_text = str(response)
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
                
                with open(DB_FILE, "w") as f:
                    json.dump(st.session_state.messages, f)

            except Exception as e:
                st.error(f"Error connecting to AI: {e}")

