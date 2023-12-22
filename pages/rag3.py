import streamlit as st
from pathlib import Path
import qdrant_client
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    download_loader,
)
from llama_index.llms import Ollama
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_hub.confluence import ConfluenceReader
import os

# Streamlit UI for input
st.title("Confluence Query Interface")

# Use secrets for sensitive information
access_token = st.secrets["CONFLUENCE_ACCESS_TOKEN"]

base_url = st.text_input("Confluence Base URL", "https://espace.agir.orange.com/")
space_key = st.text_input("Space Key", "OBSMA")
model_name = st.selectbox("Select Model", ["mistral:7b-instruct-q5_K_M", "Other Model"])
query_text = st.text_area("Enter your query", "What is OBSMA?")

# OAuth2 credentials dictionary using the access token from Streamlit secrets
oauth2_credentials = {
    "client_id": "mohammedbadr.haloua@orange.com",  # You might want to secure this as well
    "client_secret": st.secrets["CLIENT_SECRET"],
    "token": {
        "access_token": access_token,
        "token_type": "Bearer"
    }
}

# Initialize ConfluenceReader with OAuth2 credentials from Streamlit secrets
reader = ConfluenceReader(base_url=base_url, oauth2=oauth2_credentials)

# Query execution
if st.button('Run Query'):
    # Load documents using the ConfluenceReader
    documents = reader.load_data(space_key=space_key, include_attachments=True, page_status="current")

    # Set up Qdrant client and vector store
    client = qdrant_client.QdrantClient(path="./qdrant_data")
    vector_store = QdrantVectorStore(client=client, collection_name="conf_MA")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Initialize LLM model
    llm = Ollama(model=model_name)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")

    # Create the index and query engine
    index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context)
    query_engine = index.as_query_engine()

    # Query and display the response
    response = query_engine.query(query_text)
    st.write(response)

# Note: Ensure that you handle the client ID securely as well, possibly using Streamlit secrets.
