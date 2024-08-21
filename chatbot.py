import os
import getpass
from langchain.document_loaders import PyMuPDFLoader
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

def load_pdf(pdf_path):
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        print(f"PDF loaded successfully: {len(documents)} pages found.")
        return documents
    except Exception as e:
        print(f"Failed to load PDF: {e}")
        return None

def create_faiss_index(documents):
    print("Creating FAISS index. This may take a few moments...")
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    split_docs = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(split_docs, embeddings)
    print("FAISS index created successfully.")
    return vector_store

def retrieve_relevant_chunks(query, vector_store, k=3):
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

def ask_question_with_rag(query, vector_store, model):
    relevant_chunks = retrieve_relevant_chunks(query, vector_store)
    context = " ".join(relevant_chunks)
    
    print("Generating response from the model...")
    response = model.invoke([HumanMessage(content=f"{query}\n\nContext: {context}")])
    return response

def main():
    os.environ["MISTRAL_API_KEY"] = getpass.getpass(prompt="Enter your Mistral API key: ")
    model = ChatMistralAI(model="mistral-large-latest")

    print("\nWelcome to the PDF-based Chatbot!")
    pdf_path = input("Please enter the path to your PDF file: ").strip()

    documents = load_pdf(pdf_path)
    if documents is None:
        return

    vector_store = create_faiss_index(documents)

    print("\nChatbot is ready! You can ask questions based on the PDF content.")
    print("Type 'exit' to end the session.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ['exit', 'quit', 'q']:
            print("Thank you for using the chatbot. Goodbye!")
            break
        
        answer = ask_question_with_rag(query, vector_store, model)
        print(f"Bot: {answer.content}\n")

if __name__ == "__main__":
    main()
