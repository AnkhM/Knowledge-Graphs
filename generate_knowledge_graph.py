from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pyvis.network import Network
from dotenv import load_dotenv
import os
import asyncio

# Load the .env file
load_dotenv()
# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature = 0, model_name="gpt-4o")

graph_tranformer = LLMGraphTransformer(llm=llm)

# Extract graph data from input text
async def extract_graph_data(text):
    documents = [Document(page_content=text)]
    graph_documents = await graph_tranformer.aconvert_to_graph_documents(documents)
    return graph_documents


# Visualize the graph
def visualize_graph(graph_documents):
    # Create network
    net = Network(height="1200px", width="100%", directed=True, notebook=False, bgcolor="#222222", font_color="white")
    nodes = graph_documents[0].nodes
    relationships = graph_documents[0].relationships

    # Build lookup for valid nodes
    node_dict = {node.id: node for node in nodes}

    # Filter out invalid edges and collect valid node IDs
    valid_edges = []
    valid_node_ids = set()
    for rel in relationships:
        if rel.source.id in node_dict and rel.target.id in node_dict:
            valid_edges.append(rel)
            valid_node_ids.update([rel.source.id, rel.target.id])
    
    # Track which nodes are part of any relationship
    connected_node_ids = set()
    for rel in relationships:
        connected_node_ids.add(rel.source.id)
        connected_node_ids.add(rel.target.id)
    
    # Add valid nodes
    for node_id in valid_node_ids:
        node = node_dict[node_id]
        try:
            net.add_node(node.id, label=node.id, title=node.type, group=node.type)
        except:
            continue

    # Add valid edges
    for rel in valid_edges:
        try:
            net.add_Edge(rel.source.id, rel.target.id, label=rel.type.lower())
        except:
            continue
    
    # Configure physics
    net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitiationalConstant": -100,
                    "centralGravity": 0.01,
                    "springLength": 200,
                    "springConstant": 0.08
                },
                "minVelocity": 0.5,
                "solver": "forceAtlas2Based"
            }
        }
        """)
    
    output_file = "knowledge_graph.html"
    try:
        net.save_graph(output_file)
        print(f"Graph saved to {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"Error saving graph: {e}")
        return None

def generate_knowledge_graph(text):
    graph_documents = asyncio.run(extract_graph_data(text))
    net = visualize_graph(graph_documents)
    return net
