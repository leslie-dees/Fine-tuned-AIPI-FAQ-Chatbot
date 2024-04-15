import re
from langchain_community.tools.tavily_search import TavilySearchResults
import os

class GraderLLM:
    """
    A class to grade the relevance of documents to a given question within the context of AIPI.

    Attributes:
        pipe (pipeline.Pipeline): The pipeline used for text generation.
    """
    def __init__(self, pipe):
        """
        Initializes the GraderLLM with a given pipeline.

        Parameters:
            pipe (pipeline.Pipeline): The pipeline used for text generation.
        """
        self.pipe = pipe

    def grade_relevance(self, question, context):
        """
        Grades the relevance of a document to a specific question.

        Parameters:
            question (str): The question posed by the user.
            context (str): The document content to be graded.

        Returns:
            bool: True if the document is relevant, False otherwise.
        """
        # Grader system prompt
        GRADER_SYSTEM_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question. 
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
            Give a binary 'yes' or 'no' score to indicate whether the document is relevant to the question. 
            All questions are in relation to the AIPI (Artificial Intelligence for Product Innovation) program. 
            Do not answer the question itself. Only grade the relevance of the Documents context to the question.

            DOCUMENTS: {context}
            """

        BINARY_INCLUSION_PROMPT = "\nAre the provided documents somewhat relevant to the question, answer YES OR NO"

        grader_message = [
            {"role": "system", "content": GRADER_SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question + BINARY_INCLUSION_PROMPT},
            {"role": "assistant", "content": ""}
        ]

        prompt = self.pipe.tokenizer.apply_chat_template(grader_message, tokenize=False, add_generation_prompt=True)
        outputs = self.pipe(prompt, max_new_tokens=256, temperature=0.1, top_p=0.1, eos_token_id=self.pipe.tokenizer.eos_token_id, pad_token_id=self.pipe.tokenizer.pad_token_id)
        response = outputs[0]['generated_text'][len(prompt):].strip()

        # Perform regex matching to determine either positive or negative for grader
        positive_patterns = [r"\b(?:yes)\b", r"\b(?:are relevant)\b"]
        negative_patterns = [r"\b(?:no)\b", r"\b(?:are not relevant)\b"]

        positive_regex = re.compile("|".join(positive_patterns), flags=re.IGNORECASE)
        negative_regex = re.compile("|".join(negative_patterns), flags=re.IGNORECASE)

        is_positive = bool(positive_regex.search(response))
        is_negative = bool(negative_regex.search(response))

        grader_pass = is_positive and not is_negative

        return grader_pass


class WebSearchAgent:
    """
    A class to handle web search functionality for retrieving additional documents.

    Attributes:
        api_key (str): The API key for the web search service.
    """
    def __init__(self, api_key):
        """
        Initializes the WebSearchAgent with a given API key.

        Parameters:
            api_key (str): The API key for the web search service.
        """
        self.api_key = api_key

    def setup(self):
        """
        Conducts a web search query to retrieve relevant documents.

        Parameters:
            query (str): The search query.

        Returns:
            List[str]: A list of relevant document texts.
        """
        # Set the API key as an environment variable
        os.environ["TRAVILY_API_KEY"] = self.api_key
        web_search_tool = TavilySearchResults(k=3)
        return web_search_tool