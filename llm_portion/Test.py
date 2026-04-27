from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import json


load_dotenv()

MAX_REQUESTS_PER_MIN = 4
DELAY = 60

u_input = input("Please enter your question here: ")
with open("invoices.json", "r") as file:
    i_data = json.load(file)  

instructions = """
You are a helpful and friendly agent for a company that supports customers with their invoices.
Using the user input {u_input}, answer their questions related to their invoice data from {i_data}.
"""

useTemplate = PromptTemplate(
    input_variables=["u_input", "i_data"], template=instructions
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  temperature=0.2
)

chain = useTemplate | llm

response = chain.invoke(input={"u_input": u_input, "i_data": i_data})
print(response.content)