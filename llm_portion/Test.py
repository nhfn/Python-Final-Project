from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MAX_REQUESTS_PER_MIN = 4
DELAY = 60

u_input = input("Please enter your question here: ")

instructions = """
Using the user input {u_input}, answer their question.
"""

useTemplate = PromptTemplate(
    input_variables=["u_input"], template=instructions
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  temperature=1
)

chain = useTemplate | llm

response = chain.invoke(input={"u_input": u_input})
print(response.content)