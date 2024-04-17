import streamlit as st
from crag_pipeline import transform_query, retrieve_documents, grade_documents, web_search, generate_response
from setup_pipeline import setup_index, setup_embeddings_model, setup_pipeline_and_tokenizer, setup_grader, setup_web_search_agent
import os

# Initialize your components
pinecone_api_key = os.getenv('PINECONE_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')
adapter_id = os.getenv('ADAPTER_ID')

index = setup_index(pinecone_api_key)
embeddings_model = setup_embeddings_model()
pipe, tokenizer = setup_pipeline_and_tokenizer(adapter_id)
grader = setup_grader(pipe)
web_search_agent = setup_web_search_agent(tavily_api_key)

# Streamlit layout
st.title("Duke AIPI Program Chatbot")
st.sidebar.title("Chatbot Configuration")
st.sidebar.subheader("Set query parameters:")
top_k_documents = st.sidebar.slider("Select number of top documents to retrieve:", min_value=1, max_value=10, value=3)

# User interaction
user_input = st.text_input("Ask a question about the Duke AIPI Program:", "")
if st.button('Send'):
    with st.spinner('Generating response...'):
        transformed_question = transform_query(user_input, pipe)
        documents = retrieve_documents(transformed_question, embeddings_model, index, top_k=top_k_documents)
        graded_documents = grade_documents(transformed_question, documents, grader)
        combined_documents = web_search(transformed_question, graded_documents, web_search_agent)
        response = generate_response(transformed_question, combined_documents, pipe)
        st.write(response)
        st.success('Response generated!')

# Beautifying with CSS
st.markdown("""
<style>
    .css-18e3th9 {
        padding: 0.25rem 0.75rem;
        margin-bottom: 10px;
    }
    .css-1d391kg {
        padding-top: 3.5rem;
        padding-bottom: 3.5rem;
    }
</style>
""", unsafe_allow_html=True)

