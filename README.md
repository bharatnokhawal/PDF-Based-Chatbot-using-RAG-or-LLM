PDF-Based Chatbot using RAG (Retrieval-Augmented Generation)

This repository contains a Python-based chatbot that answers questions based on the content of a PDF document. The chatbot leverages Retrieval-Augmented Generation (RAG) by combining Hugging Face embeddings and a FAISS index with the Mistral API for generating responses.

Features:

PDF Content-Based Responses: The chatbot provides answers based on the content extracted from a PDF document.

Hugging Face Embeddings: Uses sentence-transformers/all-MiniLM-L6-v2 model for generating embeddings of the text.

FAISS Indexing: Efficient retrieval of relevant text chunks using FAISS (Facebook AI Similarity Search).

Mistral API: The chatbot uses the Mistral API to generate human-like responses based on the retrieved content.

Interactive CLI: A user-friendly command-line interface allows continuous interaction with the chatbot.



Getting Started:-


Prerequisites

Ensure you have the following installed:


Python 3.7 or later

Required Python packages (listed in requirements.txt)

Setup :-

Mistral API Key



You'll need a Mistral API key to use the chatbot. If you don't have one, you can get it from the Mistral API provider.

When you run the script, you'll be prompted to enter your Mistral API key.



PDF Document


Place the PDF document you want the chatbot to use in the project directory or provide the path to it when prompted.


Usage:----


Run the chatbot script: --- python chatbot.py


Interaction:----


Start the Chatbot---


When you start the script, you'll be asked to enter the path to your PDF file.



Ask Questions--


Once the PDF is loaded and the FAISS index is created, you can start asking questions based on the content of the PDF.



Exit the Chat--


Type exit, quit, or q to end the session.


Acknowledgments----

Hugging Face for the sentence-transformers/all-MiniLM-L6-v2 model.

Mistral for providing the API used to generate responses.
