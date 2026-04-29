from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent

from tools import find_highest_invoice, verify_invoice
import json

load_dotenv()

MAX_REQUESTS_PER_MIN = 4
DELAY = 60

# Wrap so that tool doesn't need arguments
@tool
def find_highest_invoice_wrapper():
    """Finds the invoice with the highest total"""
    with open("invoices.json", "r") as file:
        invoices = json.load(file) 
    return find_highest_invoice(invoices)

@tool
def verify_invoice_wrapper(i_pos: int):
    """Checks to make sure invoice is correct

    Args:
        i_pos(int): the position in the list of your invoice

    """
    with open("invoices.json", "r") as file:
        invoices = json.load(file)
    #invoices is a list of dicts, so need a number of which dict to use
    return verify_invoice(invoices[i_pos])
    #can solve by used a fix number for position

instructions = """
You are a helpful and friendly agent for a company that supports customers with their invoices.
Using the user input {u_input}, answer their questions related to their invoice data 
using these python tools {tools}.
"""

tools = [find_highest_invoice_wrapper, verify_invoice_wrapper]
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