from PyPDF2 import PdfReader
from dotenv import load_dotenv
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
from langchain.chains.question_answering import load_qa_chain
import os
from langchain import HuggingFaceHub
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
            embeddings=HuggingFaceEmbeddings()
            #embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
            
            VectorStore= FAISS.from_texts(chunks,embedding= embeddings)
            with open(f"{store_name}.pkl", "wb") as f : 
                pickle.dump(VectorStore, f)
            st.write("Embeddings computated")
        query=st.text_input("Write your query")
        st.write(query)
        if query:
            docs= VectorStore.similarity_search(query=query,k=3)
            #st.write(docs)
            repo_id = "tiiuae/falcon-7b-instruct"
            llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.7, "max_new_tokens":700})
            chain=load_qa_chain(llm=llm,chain_type="stuff")
            response=chain.run(input_documents=docs,question=query)
            st.write(response)






if __name__=='__main__':
    main()
