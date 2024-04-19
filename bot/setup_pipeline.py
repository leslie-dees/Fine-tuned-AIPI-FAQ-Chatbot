import os
import torch
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline
from utils import GraderLLM, WebSearchAgent  # Make sure to import GraderLLM from utils

def setup_index(api_key):
    """
    Initializes and returns the Pinecone index for document retrieval.
    
    Parameters:
        api_key (str): API key for Pinecone access.

    Returns:
        index: Pinecone index object.
    """
    pc = Pinecone(api_key=api_key)
    index_name = 'aipi-chatbot'
    index = pc.Index(index_name)
    return index

def setup_embeddings_model():
    """
    Loads and returns the sentence embeddings model.

    Returns:
        embeddings_model: Loaded sentence transformer model.
    """
    embeddings_model = SentenceTransformer("WhereIsAI/UAE-Large-V1")
    return embeddings_model

def setup_pipeline_and_tokenizer(adapter_id):
    """
    Initializes and returns the pipeline and tokenizer for generating responses.

    Parameters:
        adapter_id (str): Identifier for the model adapter.

    Returns:
        pipe (pipeline.Pipeline): Pipeline for text generation.
        tokenizer (AutoTokenizer): Tokenizer for the model.
    """
    # Free CUDA memory
    try:
        del model
        del tokenizer
    except:
        pass
    torch.cuda.empty_cache()

    # Determine running device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}")

    # Load model with PEFT adapter
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device,
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_id)

    # Load merged model into pipeline
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return pipe, tokenizer

def setup_grader(pipe):
    """
    Creates an instance of the GraderLLM class.

    Parameters:
        pipe (pipeline.Pipeline): Pipeline used for text generation.

    Returns:
        grader (GraderLLM): An instance of GraderLLM for grading document relevance.
    """
    grader = GraderLLM(pipe)
    return grader

def setup_web_search_agent(api_key):
    """
    Initializes and returns the web search agent for retrieving additional documents.

    Parameters:
        api_key (str): API key for the web search service.

    Returns:
        web_search_agent: Web search agent object.
    """
    web_search_agent = WebSearchAgent(api_key)
    web_search_tool = web_search_agent.setup()
    return web_search_tool