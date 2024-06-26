import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore

#from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama

from langchain.schema.output_parser import StrOutputParser

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time
import pickle


directory_path = "./pickup"
llm = Ollama(model="llama3")
embedder = OllamaEmbeddings(model="nomic-embed-text")
embedding_dimension = 768 
if "chat_history_limit" not in st.session_state:
    st.session_state.chat_history_limit = 10

load_dotenv()

def get_vectorstore():
    load_dotenv()
    
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    #PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    from pinecone import Pinecone, ServerlessSpec
    database = Pinecone(api_key=PINECONE_API_KEY)
    serverless_spec = ServerlessSpec(cloud="aws", region="us-east-1")
    INDEX_NAME = "enterprise"
    is_new_database = False
    if INDEX_NAME not in database.list_indexes().names():
        is_new_database = True
        database.create_index(
            name=INDEX_NAME,
            dimension=embedding_dimension,
            metric="cosine",
            spec=serverless_spec,
        )
    
    if is_new_database:
        print("Run the data load up to initialize Pinecone dataset")
        vector_store = PineconeVectorStore.from_documents(split_documents, embedding=embedder, index_name=INDEX_NAME)
    else: 
        vector_store =  PineconeVectorStore(index_name=INDEX_NAME, embedding=embedder)
        print("VectorStore Loaded")
        
    time.sleep(1)
    pinecone_index = database.Index(INDEX_NAME)
    
    
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOllama(model="llama3")
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOllama(model="llama3")
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

def check_history_limit():
    while len(st.session_state.chat_history) > st.session_state.chat_history_limit:
        st.session_state.chat_history.pop(0)
    return

# app config
st.set_page_config(page_title="Conversational Chat with Various Documents", page_icon="🤖")
st.title("Chat with various documents")

# sidebar
with st.sidebar:
    st.header("Set Message Queue")
    chat_history_limit = st.slider("Choose the number of messages in chat history.. ", 1, 20, 10)
    st.session_state.chat_history_limit = chat_history_limit
    st.write("Set ",st.session_state.chat_history_limit," messages in the chat history")

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore()    

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))
    check_history_limit()
    
# display conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
