import os
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ibm import WatsonxLLM

# Check GPU availability
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None


# Custom prompt for document-based question answering
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI assistant answering questions from uploaded documents.

Use ONLY the provided context to answer the question.

Guidelines:
- Be accurate and factual.
- Keep the answer directly related to the user's question.
- Do not invent information.
- Do not add unrelated explanations.
- If the answer is not found in the context, say:
  "I could not find relevant information in the uploaded document."
- Answer in a clear, professional, and easy-to-understand way.

Retrieved Context:
{context}

User Question:
{question}

Helpful Answer:
"""
)


# Function to initialize the language model and embeddings
def init_llm():
    global llm_hub, embeddings

    logger.info("Initializing WatsonxLLM and embeddings...")

    MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
    WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
    PROJECT_ID = "skills-network"

    model_parameters = {
        "max_new_tokens": 256,
        "temperature": 0.1,
    }

    # Initialize IBM watsonx.ai LLM
    llm_hub = WatsonxLLM(
        model_id=MODEL_ID,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        params=model_parameters
    )

    logger.debug("WatsonxLLM initialized: %s", llm_hub)

    # Initialize embedding model
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}
    )

    logger.debug("Embeddings initialized with model device: %s", DEVICE)


# Function to process a PDF document
def process_document(document_path):
    global conversation_retrieval_chain

    logger.info("Loading document from path: %s", document_path)

    # Load PDF
    loader = PyPDFLoader(document_path)
    documents = loader.load()

    logger.debug("Loaded %d document(s)", len(documents))

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64
    )

    texts = text_splitter.split_documents(documents)

    logger.debug("Document split into %d text chunks", len(texts))

    # Create Chroma vector database
    logger.info("Initializing Chroma vector store from documents...")

    db = Chroma.from_documents(
        texts,
        embedding=embeddings
    )

    logger.debug("Chroma vector store initialized.")

    # Optional Chroma collection debug
    try:
        collections = db._client.list_collections()
        logger.debug("Available collections in Chroma: %s", collections)
    except Exception as e:
        logger.warning("Could not retrieve collections from Chroma: %s", e)

    # Create RAG RetrievalQA chain with custom prompt
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "lambda_mult": 0.25
            }
        ),
        return_source_documents=False,
        input_key="question",
        chain_type_kwargs={
            "prompt": custom_prompt
        }
    )

    logger.info("RetrievalQA chain created successfully.")


# Function to process user question
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history

    logger.info("Processing prompt: %s", prompt)

    if conversation_retrieval_chain is None:
        return "Please upload and process a PDF document first before asking questions."

    # Run RAG chain
    output = conversation_retrieval_chain.invoke({
        "question": prompt,
        "chat_history": chat_history
    })

    answer = output["result"]

    logger.debug("Model response: %s", answer)

    # Save conversation history
    chat_history.append((prompt, answer))

    logger.debug(
        "Chat history updated. Total exchanges: %d",
        len(chat_history)
    )

    return answer


# Initialize LLM and embeddings when this file loads
init_llm()

logger.info("LLM and embeddings initialization complete.")