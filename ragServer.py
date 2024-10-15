from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from elasticsearch import Elasticsearch
import pandas as pd
import io
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Now you can access the API key like this
openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])


# Set up Elasticsearch connection
es_endpoint = "https://6e60d96a268e44dbbf4421f4a60a6701.us-central1.gcp.cloud.es.io:443"
api_key_id = "SNeqhJIBBIB8vTBV5SHk"  # Replace with your actual API key ID
api_key = "JBfO3eI1S8qDoWsmfjBtkw"  # Replace with your actual API key
es = Elasticsearch(es_endpoint, api_key=(api_key_id, api_key), verify_certs=True)

def embed_text(text):
    """Embed text using OpenAI's Embeddings API."""
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

def create_index_with_dense_vector():
    """Create Elasticsearch index with only Product Name, TID, and Price fields."""
    index_name = "product_dataset"
    mapping = {
        "mappings": {
            "properties": {
                "PRODUCT_NAME": {"type": "text"},
                "TID": {"type": "text"},
                "PRICE_RETAIL": {"type": "text"},
                "embedding": {"type": "dense_vector", "dims": 1536}  # Based on embedding model
            }
        }
    }

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=mapping)

def index_custom_data(data):
    """Index custom data with embeddings into Elasticsearch."""
    for _, row in data.iterrows():
        # Create a meaningful text representation for embedding
        text = f"Product: {row['PRODUCT_NAME']}, Price: {row['PRICE_RETAIL']}, TID: {row['tid']}"
        
        # Create the document with only the required fields
        doc = {
            'PRODUCT_NAME': row['PRODUCT_NAME'],
            'TID': row['tid'],
            'PRICE_RETAIL': row['PRICE_RETAIL'],
            'embedding': embed_text(text)  # Embed the text representation
        }
        es.index(index="product_dataset", body=doc)

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Load CSV data, embed, and index it into Elasticsearch from file upload."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        data = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
        
        create_index_with_dense_vector()  # Create the index with necessary settings
        index_custom_data(data)  # Index the data
        
        return jsonify({"message": "CSV data indexed successfully."}), 201
    else:
        return jsonify({"error": "File type not supported. Please upload a CSV file."}), 400

def perform_exact_tid_search(tid):
    """Perform exact search on Elasticsearch index using the TID field."""
    search_response = es.search(index="product_dataset", body={
        "query": {
            "term": {
                "TID": tid  # Exact match search on TID field
            }
        }
    })
    return search_response['hits']['hits']

@app.route('/ask_query', methods=['POST'])
def ask_query():
    """Ask a query and return a human-readable response for matching Product Name, TID, and Price."""
    user_query = request.json.get("query")
    
    # Check if the query is asking for a specific TID
    if "tid" in user_query.lower():
        tid_value = ''.join(filter(str.isdigit, user_query))  # Extract numeric TID value
        if tid_value:
            # Perform exact match search based on TID
            results = perform_exact_tid_search(tid_value)
            if results:
                # Extract the first matching product
                product = results[0]['_source']
                product_name = product['PRODUCT_NAME']
                price = product['PRICE_RETAIL']
                tid = product['TID']
                
                # Return a human-readable sentence response
                return jsonify({
                    "response": f"The price of {product_name} with TID {tid} is ${price}."
                })
            return jsonify({"response": f"No product found with TID {tid_value}."})
    
    # Fallback to semantic search if no exact match is found for TID
    return jsonify({"response": "Please provide a valid TID in your query."})

if __name__ == "__main__":
    app.run(debug=True)
