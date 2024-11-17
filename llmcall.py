import base64
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import TransformChain
import networkx as nx
import matplotlib.pyplot as plt
import os

#Add the key here:
os.environ["OPENAI_API_KEY"] = "your_key"

def encode_image(image_path):
    """Encode an image as base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_image(inputs: dict) -> dict:
    """Load image from file and encode it as base64."""
    return {"image": encode_image(inputs["image_path"])}

load_image_chain = TransformChain(
    input_variables=['image_path'],
    output_variables=["image"],
    transform=load_image
)

class Triple(BaseModel):
    """Triple class to extract content in a structured way."""
    subject: str = Field(description="Subject of the triple.")
    relation: str = Field(description="Describes relation between subject and object.")
    objectt: str = Field(description="Object of the triple.")

parser = JsonOutputParser(pydantic_object=Triple)

@chain
def image_model(inputs: dict):
    """Invoke model with image and prompt."""
    model = ChatOpenAI(model="gpt-4o", temperature=0.01)
    prompt_content = [
        {"type": "text", "text": inputs["prompt"]},
        {"type": "text", "text": parser.get_format_instructions()},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image']}"}} 
    ]
    
    msg = model.invoke([HumanMessage(content=prompt_content)])
    return msg.content

def get_triple_info(image_path: str) -> list:
    """Extract triples from an image."""
    processed_triples = load_image_chain | image_model | parser

    prompt = """
    Analyze the diagram and extract triples in the format of 'subject relation object'.
    Guidelines:
    - Arrows denote direction: subject to object.
    - For solid arrows, use the specified relation; default to “has” if unspecified.
    - Consider dotted arrows as direct subject-object connections, bypassing intermediary nodes.
    - Do not infer any relationships beyond what the arrows explicitly depict.
    Ensure that each triple directly represents the arrows' start and end points.
    Only return the triples without additional text.
    """

    return processed_triples.invoke({'image_path': image_path, 'prompt': prompt})

def visualize_triples(triples: list):
    """Visualize triples as a directed graph using networkx and matplotlib."""
    graph = nx.DiGraph()

    for triple in triples:
        graph.add_node(triple['subject'])
        graph.add_node(triple['objectt'])
        graph.add_edge(triple['subject'], triple['objectt'], label=triple['relation'])

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10)
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    plt.title("Graph Representation of Triples")
    plt.show()

def main():
    """Main function to execute the script logic."""
    triple_info = get_triple_info("test1.png")
    print(triple_info)
    visualize_triples(triples=triple_info)

if __name__ == "__main__":
    main()