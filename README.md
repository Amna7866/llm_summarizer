# llm_summarizer
This project is a document summarization tool built using open-source language models (LLMs) that run entirely on your local CPU, with no need for OpenAI APIs. Users can upload PDF files, and the app will extract and summarize the text using Hugging Face's transformers and langchain libraries. It features a simple Streamlit GUI and a backend built for efficient offline text processing.

Steps to Run the Application
Clone the Repository
Set Up Virtual Environment
Once the setup is complete, you can run the Streamlit application: streamlit run app.py




Features
PDF Upload & Parsing: Upload multi-page PDFs; text is auto-extracted using PyPDFLoader.
Summarize documents using open-source LLMs like T5
Streamlit Interface: Clean, interactive frontend for uploading, viewing, and summarizing documents.

Design Overview
Text Splitting with LangChain
Large PDF documents are split into manageable chunks using RecursiveCharacterTextSplitter, improving summarization accuracy.
Summarization with Transformers
The summarization is powered by Hugging Face’s T5 model (T5ForConditionalGeneration) and pipeline, running efficiently on CPUs using torch.

Technologies Used
Python
Streamlit: GUI framework
LangChain: For document loaders and text chunking
Transformers (Hugging Face): Summarization model
Torch: Backend support for model inference
PyPDFLoader: Extract text from PDFs
