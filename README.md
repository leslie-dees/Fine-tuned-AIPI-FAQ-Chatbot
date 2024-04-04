# Fine-tuned-AIPI-FAQ-Chatbot

## TODO
* Select & fine-tune opensource model for chat capabilities
    * Select commonly used chat/instruct dataset for fine-tuning
        * https://huggingface.co/datasets/hakurei/open-instruct-v1
    * Host on HuggingFace
* Scrape FAQ, domain page, domain sublinks for raw text data
    * Also utilize commonly asked questions at the bottom of this website:
        * https://sites.duke.edu/aipi/new-student-resources/
    * Host data in vector db (Pinecone?)
* Build RAG pipeline using fine-tuned model and AIPI data
    * Frontend display
* Model hosting