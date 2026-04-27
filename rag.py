import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Configuration & API Setup
os.environ["GOOGLE_API_KEY"] = "AIzaSyBTHhIPi2StwZLIQcK9Niq55VSyquPF5bI"

def initialize_vector_db(invoice_data: list[str], collection_name="invoice_vault"):
    """
    Processes raw text, embeds it, and stores it in a ChromaDB collection.
    """
    # Initialize Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="retrieval_document")
    # Convert raw text to LangChain Documents
    docs = [Document(page_content=text, metadata={"source": "invoice_upload"}) for text in invoice_data]

    # Split text into manageable chunks for better retrieval precision
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # Create and persist the vector database
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

# 2. Retrieval Function
def retrieve_invoice_context(query: str, vectorstore):
    """
    Searches the vector DB for the most relevant invoice snippets.
    """
    results = vectorstore.similarity_search(query, k=2)
    
    context = "\n---\n".join([res.page_content for res in results])
    return context

# 3. Testing Logic
if __name__ == "__main__":
    # Mock Invoice Data
    mock_invoices = [
        "Invoice #1001: Client: Acme Corp. Date: 2024-03-01. Total: $1,500.00. Items: 10x Cloud Storage Units, 2x Consulting Hours.",
        "Invoice #1002: Client: Stark Ind. Date: 2024-03-05. Total: $50,000.00. Items: Arc Reactor Maintenance, Vibranium Polishing.",
        "Invoice #1003: Client: Wayne Ent. Date: 2024-03-10. Total: $12,450.00. Items: Tactical Suit Upgrades, Bat-Signal Bulbs."
    ]

    print("--- Step 1: Embedding and Storing Data ---")
    vdb = initialize_vector_db(mock_invoices)
    print("Database initialized successfully.")

    print("\n--- Step 2: Testing Retrieval ---")
    test_query = "How much did we charge Wayne Ent for tactical upgrades?"
    retrieved_info = retrieve_invoice_context(test_query, vdb)

    print(f"Query: {test_query}")
    print(f"Retrieved Context:\n{retrieved_info}")

    # Validation check
    if "Wayne Ent" in retrieved_info and "$12,450.00" in retrieved_info:
        print("\nTest Result: PASS")
    else:
        print("\nTest Result: FAIL")