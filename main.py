from PyPDF2 import PdfReader
from dotenv import load_dotenv
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import os

load_dotenv()


def main():
    st.header("Chat with pdf")
    pdf=st.file_uploader("Upload PDF", type='pdf')
    if pdf is not None:
        reader = PdfReader(pdf)
        number_of_pages = len(reader.pages)
        st.write("Taking a bit long to process ",number_of_pages,"pages")
        text = ""
        for page in reader.pages:
            text+= page.extract_text() 
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text=text)
        store_name = pdf.name[ : -4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f : 
                VectorStore= pickle.load(f)
            st.write("Embeddings loaded from the disk")
        else:
            embeddings=OpenAIEmbeddings()
            VectorStore= FAISS.from_texts(chunks,embedding= embeddings)
            with open(f"{store_name}.pkl", "wb") as f : 
                pickle.dump(VectorStore, f)
            st.write("Embeddings computated")





if __name__=='__main__':
    main()