
# streamlit is a powerful Python library that allows us to build interactive web applications with minimal effort. 
# We'll use it to create the user interface for our LLM app
from http import client
from pydoc import cli
from ollama import embeddings
import streamlit as st

# tempfile is a Python standard library module that provides functionality for creating temporary files and directories. 
# We'll use it to set up a temporary location for our vector database.
import tempfile


from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb
from agno.knowledge.embedder.ollama import OllamaEmbedder



# Define the collection name for the vector database
collection_name = "pdf-reader-index"
ef = OllamaEmbedder()

# Set up Qdrant as the vector database with the embedder
vector_db = ChromaDb(
    collection=collection_name,
    path="tmp/chromadb", 
    persistent_client=True,
    embedder=ef
)

# Define the knowledge base
knowledge_base = Knowledge(
    vector_db=vector_db,
)

# #clear the vector database
# chroma_client = knowledge_base.vector_db._client
# all_collections = chroma_client.list_collections()
# for col in all_collections:
#     chroma_client.delete_collection(name=col.name)


# # Add content to the knowledge base, comment out after the first run to avoid reloading
knowledge_base.add_content(
    path=r"C:\Users\SAGAR SANGVEKAR\OneDrive\Documents\RahulKumarResume.pdf"
)
print("Content added to the knowledge base.")   

# chroma_client = knowledge_base.vector_db._client
# all_collections = chroma_client.list_collections()
# for col in all_collections:
#     collectiontmp = chroma_client.get_collection(name=col.name)
#     print(col.name, collectiontmp.count())
#     print(collectiontmp.peek());
# # Set up the Streamlit App

st.title("Chat with Your PDFs")
st.caption("A locally hosted LLM app with RAG for conversing with your PDF documents.")

#Create a temporary directory for the vector database
temp_dir = tempfile.mkdtemp()


uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    # Create a temporary file and write the contents of the uploaded file to it
    print("File uploaded successfully.")
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    # Add the PDF file to the knowledge base
    # Add content to the knowledge base, comment out after the first run to avoid reloading
    knowledge_base.add_content(
        temp_file_path
    )
    st.success(f"Successfully added to the knowledge base!")
    
    # results = knowledge_base.search(query="SQL")

    # for doc in results:
    #     print(f"Found: {doc.content[:100]}...")  # Print first 100 chars of each match


chroma_client = knowledge_base.vector_db._client
all_collections = chroma_client.list_collections()
for col in all_collections:
     collectiontmp = chroma_client.get_collection(name=col.name)
     print(col.name, collectiontmp.count())
     print(collectiontmp.peek());

# # Create the Agent using Ollama's llama3.2 model and the knowledge base
agent = Agent(
    name="Local RAG Agent",
    model=Ollama(id="llama3.2"),
    knowledge=knowledge_base,
)

question = st.text_input("Ask a question about the PDF:")
if question:
    try:
        answer = agent.print_response(question)
        st.info(answer)
    except Exception as e:
        st.error(f"An error occurred: {e}")