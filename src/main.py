from setup_pipeline import setup_index, setup_embeddings_model, setup_pipeline_and_tokenizer, setup_grader, setup_web_search_agent
from crag_pipeline import transform_query, retrieve_documents, grade_documents, web_search, generate_response
import os
import warnings
warnings.filterwarnings("ignore")

# Setup the CRAG pipeline
pinecone_api_key = os.getenv('PINECONE_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')
adapter_id = os.getenv('ADAPTER_ID')
index = setup_index(pinecone_api_key)
embeddings_model = setup_embeddings_model()
pipe, _ = setup_pipeline_and_tokenizer(adapter_id)
web_search_agent = setup_web_search_agent(tavily_api_key)

def main():
    """
    Main function to run the chatbot for the Duke AIPI program.
    """
    # Example user question
    user_question = "What are some restaurants?"

    # Process flow
    transformed_question = transform_query(user_question, pipe)
    documents = retrieve_documents(transformed_question, embeddings_model, index)
    if documents:
        # If documents retrieved with >0.5 cosine similarity
        response = generate_response(transformed_question, documents, pipe)
    else:
        # If no documents retrieved, use websearch to find the answer
        documents_w_websearch  = web_search(transformed_question, documents, web_search_agent)
        response = generate_response(transformed_question, documents_w_websearch, pipe)

    print("Generated Response:")
    print(response)
    return response

if __name__ == "__main__":
    main()
