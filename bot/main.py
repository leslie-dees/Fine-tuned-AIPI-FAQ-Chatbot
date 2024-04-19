import os
import time
import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from setup_pipeline import setup_index, setup_embeddings_model, setup_pipeline_and_tokenizer, setup_grader, setup_web_search_agent
from crag_pipeline import transform_query, retrieve_documents, grade_documents, web_search, generate_response

import warnings
warnings.filterwarnings("ignore")

"""
ENV VARS
"""
def load_env():
    load_dotenv()
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    adapter_id = os.getenv('ADAPTER_ID')
    return pinecone_api_key, tavily_api_key, adapter_id

"""
RUNTIME SETUP
"""
def setup():
    print("Loading env vars..")
    pinecone_api_key, tavily_api_key, adapter_id = load_env()
    print("Configuring app..")
    app = FastAPI()
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")
    print("Loading pipeline components..")
    index = setup_index(pinecone_api_key)
    embeddings_model = setup_embeddings_model()
    pipe, _ = setup_pipeline_and_tokenizer(adapter_id)
    web_search_agent = setup_web_search_agent(tavily_api_key)
    print("Chatbot ready...")
    return app, index, embeddings_model, pipe, web_search_agent

app, index, embeddings_model, pipe, web_search_agent = setup()

"""
ENDPOINT CONFIG
"""
# Landing Route
@app.get("/")
def landing():
    return FileResponse("index.html")

# Classifier Route
@app.post("/chat")
async def chat(request_data: dict):
    try:
        conversation = request_data.get("conversation")
        user_question = conversation[-1]["text"] if conversation else ""
        # Generate response
        documents = retrieve_documents(user_question, embeddings_model, index)
        if documents:
            # If documents retrieved with >0.5 cosine similarity
            response = generate_response(user_question, documents, pipe)
        else:
            # If no documents retrieved, use websearch to find the answer
            transformed_question = transform_query(user_question, pipe)
            documents_w_websearch  = web_search(transformed_question, documents, web_search_agent)
            response = generate_response(transformed_question, documents_w_websearch, pipe)
            print(response)
        return {"body": response}
    except Exception as e:
        print(e)
        return {"body": "Sorry, I didn't get that."}