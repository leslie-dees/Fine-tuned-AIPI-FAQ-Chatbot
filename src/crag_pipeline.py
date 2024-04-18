from typing import List, Dict
from setup_pipeline import setup_embeddings_model, setup_index  # Ensure these are imported from your setup module
from utils import GraderLLM, WebSearchAgent  # Import GraderLLM from utils if not already defined in the same file

def transform_query(question: str, pipe) -> str:
    """
    Transforms the input question to a better version optimized for retrieval understanding and web search.

    Parameters:
        question (str): The original user question.
        pipe (pipeline.Pipeline): The pipeline for text generation.

    Returns:
        str: The transformed question.
    """
    RE_WRITER_SYSTEM = """
    You are question re-writer that converts an input question to a better version that is optimized for web search for the Duke Artificial Intelligence for Product Innovation (AIPI) program.
    Look at the input and try to reason about the underlying semantic intent / meaning.
    Make sure to include the Duke University Master's in Artificial Intelligence program in the new question.
    """

    re_write_prompt = [
        {"role": "system", "content": RE_WRITER_SYSTEM},
        {"role": "user", "content": "Here is the initial question: \n\n {question} \n Formulate an improved question".format(question=question)}
    ]

    re_write_prompt = pipe.tokenizer.apply_chat_template(re_write_prompt, tokenize=False, add_generation_prompt=True)
    re_write_outputs = pipe(re_write_prompt, max_new_tokens=256, temperature=0.1, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    re_write_response = re_write_outputs[0]['generated_text'][len(re_write_prompt):].strip()

    return re_write_response

def retrieve_documents(question: str, embeddings_model, index) -> List[str]:
    """
    Retrieves documents relevant to the transformed question using Pinecone vector search.

    Parameters:
        question (str): The transformed question for which documents need to be retrieved.
        embeddings_model: The sentence embedding model used for creating query vectors.
        index: The Pinecone index to query.

    Returns:
        List[str]: A list of retrieved document texts.
    """
    question_embeddings = embeddings_model.encode(question)

    matches = index.query(
        vector=question_embeddings.tolist(),
        top_k=2,  # Adjust based on how many documents you want to retrieve
        include_metadata=True
    )
    documents = [match['metadata']['text'] for match in matches['matches'] if match['metadata']['score'] >= 0.5]
    return documents

def grade_documents(question: str, documents: List[str], grader: GraderLLM) -> List[str]:
    """
    Filters the retrieved documents to only include those that are relevant to the question.

    Parameters:
        question (str): The user's question.
        documents (List[str]): A list of retrieved document texts.
        grader (GraderLLM): The document relevance grader.

    Returns:
        List[str]: A list of filtered documents that are deemed relevant.
    """
    filtered_docs = [doc for doc in documents if grader.grade_relevance(question, doc)]
    return filtered_docs

def web_search(question: str, documents: List[str], web_search_agent:WebSearchAgent) -> List[str]:
    """
    Retrieves additional documents using a web search agent and combines them with the retrieved documents.

    Parameters:
        question (str): The user's question.
        documents (List[str]): A list of retrieved document texts.
        web_search_agent: The web search agent for retrieving additional documents.

    Returns:
        List[str]: A combined list of retrieved and web-searched document texts.
    """

    docs = web_search_agent.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    documents.append(web_results)
    return documents

def generate_response(question: str, documents: List[str], pipe) -> str:
    """
    Generates a response based on the relevant documents and the user's question.

    Parameters:
        question (str): The user's question.
        documents (List[str]): A list of relevant document texts.
        pipe (pipeline.Pipeline): The pipeline for text generation.

    Returns:
        str: The generated text response.
    """
    formatted_documents = "\n".join(document for document in documents)
    SYSTEM_PROMPT = """
    You are a helpful AI assistant. Users will ask you questions. 
    Take a moment to think then respond with a polite and 
    appropriate answer. You may use the provided CONTEXT if it 
    is useful and improves your response. If you are unsure of the 
    answer, you can respond "I don't know." or "I'm not sure.".

    CONTEXT: This is in context of the Duke Artificial Intelligence for Product Innovation (AIPI) Masters's program. {context}
    """

    message = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context = formatted_documents)},
        {"role": "user", "content": question},
    ]

    prompt = pipe.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, temperature=0.1, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    generation = outputs[0]['generated_text'][len(prompt):].strip()
    return generation
