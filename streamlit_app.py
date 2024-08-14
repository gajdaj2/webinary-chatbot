import streamlit as st

from chromadb_check import chroma_collection
from ollama_model import ollama_response

st.title("Prosty chatbot")
pytanie = st.text_input("Wpisz pytanie")
generuj = st.button("Generuj odpowiedz")

if generuj:
    with (st.spinner("Generuje odpowiedz")):
        results = chroma_collection.query(query_texts=[pytanie], n_results=5)
        retrieved_documents = results['documents'][0]
        pytanie_z_contextem = (f"Answer question {pytanie} "
                               f"base your answer on text bellow. Text: \n {retrieved_documents}")
        st.text(ollama_response(pytanie_z_contextem))