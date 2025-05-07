import streamlit as st  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.chains.summarize import load_summarize_chain 
from transformers import T5Tokenizer, T5ForConditionalGeneration 
from transformers import pipeline
import torch 
import base64 

#MODEL AND TOKENIZER 
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint) 
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map = 'auto', torch_dtype = torch.float32) 

#FILE LOADER AND PREPROCESSING
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    
    final_texts = " "
    for text in texts:
        print(text)
        final_texts += text.page_content
    
    return final_texts