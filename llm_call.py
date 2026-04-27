from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
import json
from tools import calculate_total, verify_invoice, find_highest_invoice, find_invoices_by_vendor


load_dotenv()

MAX_REQUESTS_PER_MIN = 4
DELAY = 60

tools = [calculate_total, verify_invoice, find_highest_invoice, find_invoices_by_vendor]

u_input = input("Please enter your question here: ")
# with open("invoices.json", "r") as file:
#     i_data = json.load(file)  

instructions = """
You are a helpful and friendly agent for a company that supports customers with their invoices.
Using the user input {u_input}, answer their questions related to their invoice data using these python tools {tools}.
"""

useTemplate = PromptTemplate(
    input_variables=["u_input", "tools"], template=instructions
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  temperature=0.2
)

chain = useTemplate | llm


response = chain.invoke(input={"u_input": u_input, "tools": tools})
print(response.content)