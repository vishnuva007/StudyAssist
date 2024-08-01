from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from PyPDF2 import PdfReader
import openai
import os

from prompts import PROMPT_QUESTIONS

# Load Azure OpenAI API key and endpoint from environment variables
azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

# Initialize OpenAI client for Azure
openai.api_key = azure_openai_api_key
openai.api_base = azure_openai_endpoint
openai.api_version = "2024-02-01"  

# Function to load data from PDF
def load_data(uploaded_file):
    # Load data from PDF
    pdf_reader = PdfReader(uploaded_file)
    # Combine text from Document into one string
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

# Function to split text into chunks
def split_text(text, chunk_size, chunk_overlap):
    # Initialize text splitter
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Split text into chunks
    text_chunks = text_splitter.split_text(text)

    # Convert chunks to documents
    documents = [Document(page_content=t) for t in text_chunks]

    return documents

# Function to initialize large language model
def initialize_llm(openai_api_key, model, temperature):
    # Define a function to query the LLM
    def query_llm(prompt):
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message["content"].strip()

    return query_llm

# Function to generate questions
def generate_questions(llm, chain_type, documents):
    questions = ""
    for document in documents:
        prompt = PROMPT_QUESTIONS.format(text=document.page_content)
        questions += llm(prompt) + "\n"
    return questions

# Function to create Retrieval QA Chain
def create_retrieval_qa_chain(openai_api_key, documents):
    # Set embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create vector database
    vector_database = Chroma.from_documents(documents=documents, embedding=embeddings)

    # Define a function to answer questions
    def answer_question(question):
        prompt = f"Q: {question}\nA:"
        response = openai.Completion.create(
            model="gpt-4-32k",  # Update to your model
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].text.strip()

    return answer_question
