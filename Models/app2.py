import json
import os
import sys
import boto3  ##Biblioteca AWS SDK para Python, usada para interagir com os serviços da AWS.
import streamlit as st   ##biblioteca que cria interface web


# Utilizaremos o modelo Titan Embeddings para gerar embeddings

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Ingestão de Dados

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Embeddings Vetoriais e Armazenamento Vetorial

from langchain.vectorstores import FAISS

# Modelos de LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

#Configura o cliente AWS Bedrock para gerar embeddings usando o modelo amazon.titan-embed-text-v1.
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Função para ingestão de dados: data_ingestion carrega documentos PDF da pasta data e os divide em pedaços menores 
# usando RecursiveCharacterTextSplitter.
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # Em nossos testes, a divisão por caracteres funcionou melhor com este conjunto de PDFs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    
    docs = text_splitter.split_documents(documents)
    return docs

# Função para obter o armazenamento vetorial: get_vector_store gera embeddings vetoriais dos documentos 
# e os armazena localmente usando FAISS.
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# Função para criar e retornar o modelo LLM Claude
def get_claude_llm():
    llm = Bedrock(model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={'maxTokens': 512})
    return llm

# Função para criar e retornar o modelo LLM Llama2
def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama2-70b-chat-v1", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

# Template do prompt para as perguntas e respostas dos modelo
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

# Instanciação do template do prompt
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Função para obter a resposta do LLM: get_response_llm usa um modelo LLM e um armazenamento vetorial para responder a uma consulta.
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# Função principal main configura a interface do Streamlit, permitindo ao usuário fazer perguntas e atualizar o armazenamento vetorial.
def main():
    st.set_page_config("Chat PDF")
    
    st.header("Converse com PDF usando AWS Bedrock💁")

    user_question = st.text_input("Faça uma pergunta sobre os arquivos em PDF")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama2_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

# Execução do script principal
if __name__ == "__main__":
    main()
