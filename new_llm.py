from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent

from tools import find_highest_invoice
import json

load_dotenv()

# Wrap so that tool doesn't need arguments
@tool
def find_highest_invoice_wrapper():
    """Finds the invoice with the highest total"""
    with open("invoices.json", "r") as file:
        invoices = json.load(file) 
    return find_highest_invoice(invoices)

instructions = """
You are a helpful and friendly agent for a company that supports customers with their invoices.
Using the user input {u_input}, answer their questions related to their invoice data using these python tools {tools}.
"""

tools = [find_highest_invoice_wrapper]
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

agent = create_agent(
    tools = tools,
    model = llm,
    system_prompt = instructions
)

u_input = input("Please enter your question here: ")

response = agent.invoke({
"messages": [{"role": "user", "content": u_input}]
})
print(response["messages"][-1].content)