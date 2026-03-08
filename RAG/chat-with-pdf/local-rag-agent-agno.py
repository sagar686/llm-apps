# Import necessary libraries
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.os import AgentOS

# Define the collection name for the vector database
collection_name = "thai-recipe-index"

# Set up ChromaDb as the vector database with the embedder
vector_db = ChromaDb(
    collection=collection_name,
    path="tmp/chromadb", 
    persistent_client=True,
    embedder=OllamaEmbedder()
)

# Define the knowledge base
knowledge_base = Knowledge(
    vector_db=vector_db,
)

# Add content to the knowledge base, comment out after the first run to avoid reloading
# knowledge_base.add_content(
#     url="https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"
# )

# Create the Agent using Ollama's llama3.2 model and the knowledge base
agent = Agent(
    name="Local RAG Agent",
    model=Ollama(id="llama3.2"),
    knowledge=knowledge_base,
)

# UI for RAG agent
agent_os = AgentOS(agents=[agent])
app = agent_os.get_app()

# Run the AgentOS app
if __name__ == "__main__":
    agent_os.serve(app="local-rag-agent-agno:app", reload=True)