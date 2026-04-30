import os
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import find_highest_invoice, verify_invoice, calculate_total, find_invoices_by_vendor
from rag import retrieve_invoice_context

load_dotenv()

MAX_REQUESTS_PER_MIN = 4
DELAY = 60

# --- Tool Wrappers ---

@tool
def find_highest_invoice_wrapper() -> str:
    """Finds the invoice with the highest total amount."""
    try:
        with open("invoices.json", "r") as file:
            invoices = json.load(file) 
        return str(find_highest_invoice(invoices))
    except Exception as e:
        return f"Error: {e}"

@tool
def verify_invoice_wrapper(input_data: str) -> str:
    """Deterministic math verification. You MUST pass the EXACT JSON details of an invoice to verify."""
    try:
        clean_json = re.sub(r'```json|```', '', input_data).strip()
        invoice_dict = json.loads(clean_json)
        
        if "items" not in invoice_dict and len(invoice_dict) == 1:
            first_key = list(invoice_dict.keys())[0]
            invoice_dict = invoice_dict[first_key]
            
        result = verify_invoice(invoice_dict)
        return f"Status: {result['status']}\nMessage: {result['message']}"
    except Exception as e:
        return f"Tool Error: {str(e)}. Please try sending the raw JSON data."

@tool
def find_invoices_by_vendor_wrapper(vendor_name: str) -> str:
    """Finds all invoices for a given vendor name."""
    try:
        with open("invoices.json", "r") as f:
            invoices = json.load(f)
        return str(find_invoices_by_vendor(invoices, vendor_name))
    except Exception as e:
        return f"Error: {e}"

# --- Agent Factory ---

def get_invoice_agent(global_vectorstore):
    """
    Constructs and returns the LangChain AgentExecutor.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0
    )

    # We dynamically pass the vectorstore to the retrieve tool
    @tool
    def retrieve_invoice_data_bound(query: str) -> str:
        """Search the vector database for invoice details. Useful when you need details about specific invoices."""
        if not global_vectorstore:
            return "Error: Database not loaded."
        return retrieve_invoice_context(query, global_vectorstore)

    tools = [
        retrieve_invoice_data_bound, 
        verify_invoice_wrapper, 
        find_highest_invoice_wrapper, 
        find_invoices_by_vendor_wrapper
    ]

    instructions = (
        "You are a professional auditor. "
        "You have access to several tools to search and verify invoices. "
        "ALWAYS use retrieve_invoice_data_bound if you need context. "
        "To check math, pass the EXACT JSON details to verify_invoice_wrapper. "
        "Always respond in plain text, as you are a chat bot, and cant output fancy outputs."
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=instructions
    )
    return agent
