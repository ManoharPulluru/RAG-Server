from flask import Flask, request, jsonify
from flask_cors import CORS
from elasticsearch import Elasticsearch
import pandas as pd
import io
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the Gemini API key from the environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://rag-client.netlify.app"])

# Set up Elasticsearch connection
es_endpoint = "https://6e60d96a268e44dbbf4421f4a60a6701.us-central1.gcp.cloud.es.io:443"
api_key_id = "SNeqhJIBBIB8vTBV5SHk"  # Replace with your actual API key ID
api_key = "JBfO3eI1S8qDoWsmfjBtkw"  # Replace with your actual API key
es = Elasticsearch(es_endpoint, api_key=(api_key_id, api_key), verify_certs=True)

# Gemini API Key

def create_index_with_dense_vector():
    """Create Elasticsearch index with TID, PRICE_RETAIL, PRODUCT_NAME, and URL fields."""
    index_name = "product_dataset"
    mapping = {
        "mappings": {
            "properties": {
                "PRODUCT_NAME": {"type": "text"},
                "TID": {"type": "text"},
                "PRICE_RETAIL": {"type": "text"},
                "URL": {"type": "text"},
                "embedding": {"type": "dense_vector", "dims": 1536}  # Based on embedding model
            }
        }
    }

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=mapping)

def index_custom_data(data):
    """Index custom data into Elasticsearch."""
    for _, row in data.iterrows():
        # Create the document with necessary fields
        doc = {
            'PRODUCT_NAME': row['PRODUCT_NAME'],
            'TID': row['tid'],
            'PRICE_RETAIL': row['PRICE_RETAIL'],
            'URL': row['url'] if 'url' in row else ''
        }
        es.index(index="product_dataset", body=doc)

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Load CSV data and index it into Elasticsearch from file upload."""
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

def perform_fuzzy_search(query_text):
    """Perform a fuzzy search on Elasticsearch index."""
    search_response = es.search(index="product_dataset", body={
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["PRODUCT_NAME", "TID", "PRICE_RETAIL", "URL"],
                "fuzziness": "AUTO"
            }
        }
    })
    return search_response['hits']['hits']

def generate_gemini_response(input_text):
    """Generate response from Gemini API."""
    prompt = f"Reply in a professional manner: {input_text}"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            candidates = result.get("candidates", [])
            if candidates:
                content_parts = candidates[0].get("content", {}).get("parts", [])
                if content_parts:
                    gemini_response = content_parts[0].get("text", "").strip()
                    return gemini_response if gemini_response else "No content returned from the API."
                return "No content parts in the API response."
            return "No candidates in the API response."
        return f"Error calling Gemini API: {response.text}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"

@app.route('/ask_query', methods=['POST'])
def ask_query():
    """Ask a query and return a tailored response based on the specific question asked."""
    user_query = request.json.get("query").lower()  # Convert to lowercase for easier matching
    
    # Check if the query is asking for a specific TID
    if "tid" in user_query:
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
                url = product['URL']
                
                # Tailored response based on the type of question
                if "price" in user_query:
                    return jsonify({
                        "response": f"The price of TID {tid} is ${price}."
                    })
                elif "product" in user_query or "name" in user_query:
                    return jsonify({
                        "response": f"The product name of TID {tid} is {product_name}."
                    })
                elif "url" in user_query:
                    return jsonify({
                        "response": f"The URL for the product with TID {tid} is {url}."
                    })
                else:
                    # Default response if no specific type of question is detected
                    return jsonify({
                        "response": f"The product {product_name} (TID {tid}) is priced at ${price}. You can find more details here: {url}."
                    })
    
    # Fallback to fuzzy search or Gemini API for generating a response
    fuzzy_results = perform_fuzzy_search(user_query)
    if fuzzy_results:
        product = fuzzy_results[0]['_source']
        product_name = product['PRODUCT_NAME']
        price = product['PRICE_RETAIL']
        tid = product['TID']
        url = product['URL']
        return jsonify({
            "response": f"The product {product_name} (TID {tid}) is priced at ${price}. More info: {url}."
        })
    
    # If no match is found in Elasticsearch, fallback to Gemini API
    gemini_response = generate_gemini_response(user_query)
    return jsonify({"response": gemini_response})

if __name__ == "__main__":
    app.run(debug=True)
