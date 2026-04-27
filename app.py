import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from rag_pipeline import get_retriever
from tools import calculate_total, verify_invoice

# --- 1. Agent Tools Configuration ---
# These functions allow the LLM to interact with our private data and Python logic.

@tool
def retrieve_invoice_data(query: str) -> str:
    """Search the vector database for specific invoice information."""
    retriever = get_retriever()
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results])

@tool
def verify_invoice_tool(invoice_json_string: str) -> str:
    """Custom Python tool to perform deterministic math verification on invoice data."""
    import json
    invoice_dict = json.loads(invoice_json_string)
    result = verify_invoice(invoice_dict)
    return f"Status: {result['status']}"

# --- 2. Interface Design ---
# Building the interactive web dashboard using Streamlit.

st.title("📊 Invoice Analyzer Agent")
st.markdown("Ask me to analyze, retrieve, or verify invoices.")

# Handle persistent chat session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages to the UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. AI System Orchestration ---
# Configuring the Gemini LLM and the LangChain Agent.

# Initialize the generative model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# Register the tools the agent is authorized to use
agent_tools = [retrieve_invoice_data, verify_invoice_tool]

# Create the agent with a system prompt to define its business role
agent = create_agent(
    model=llm,
    tools=agent_tools,
    system_prompt="You are an AI assistant designed to help with niche business logic. You have access to tools that retrieve invoice data and verify invoice math. Use them when needed to answer the user's request accurately."
)

# --- 4. Chat Input & Processing ---
# Managing user input and generating agentic responses.

if user_prompt := st.chat_input("E.g., Can you verify the totals for INV002?"):
    # Record and display the user's prompt
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            # Prepare chat history for the agent to maintain context
            formatted_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            
            # Execute the agent workflow
            response = agent.invoke({"messages": formatted_messages})
            
            # Parse the response to ensure only the final text is displayed to the user
            if isinstance(response, list) and len(response) > 0:
                output_text = response[0].get('text', str(response))
            elif isinstance(response, dict):
                output_text = response.get("output")
                if not output_text and "messages" in response:
                    output_text = response["messages"][-1].content
            else:
                output_text = str(response)
                
            st.markdown(output_text)
            
    # Save the assistant's response to session history
    st.session_state.messages.append({"role": "assistant", "content": output_text})
