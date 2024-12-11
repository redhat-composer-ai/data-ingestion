import weaviate
import os
from langchain_community.llms import VLLMOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_weaviate_client():
    """Get the Weaviate client."""
    logger.info("Fetching Weaviate client...")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    weaviate_host = os.getenv("WEAVIATE_HOST")
    weaviate_port = os.getenv("WEAVIATE_PORT")

    if not all([weaviate_api_key, weaviate_host, weaviate_port]):
        logger.error("Weaviate config not present. Check weaviate host, port, and API key.")
        exit(1)

    auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key)
    return weaviate.Client(
        url=f"{weaviate_host}:{weaviate_port}",
        auth_client_secret=auth_config,
    )

def get_top_records(index_name, limit=10):
    """
    Fetch the top N records from a specific index in Weaviate.
    
    Args:
        weaviate_client: The Weaviate client instance.
        index_name (str): The name of the index (class) to query.
        limit (int): The maximum number of records to fetch (default is 10).
    
    Returns:
        List of records or an error message if the operation fails.
    """
    logger.info(f"Fetching top {limit} records for index: {index_name}")
    weaviate_client = get_weaviate_client()
    query = f"""
    {{
        Get {{
            {index_name} (
                limit: {limit}
            ) {{
                page_content
                metadata
            }}
        }}
    }}
    """
    
    try:
        response = weaviate_client.query.raw(query)
        data = response.get("data", {}).get("Get", {}).get(index_name, [])
        
        if not data:
            logger.warning(f"No records found for index: {index_name}")
            return []
        
        logger.info(f"Successfully fetched {len(data)} records from index: {index_name}")
        return data
    except Exception as e:
        logger.error(f"Error fetching records for index {index_name}: {e}")
        return []


def list_collections():
    """List all collections (classes) in the Weaviate instance."""
    client = get_weaviate_client()
    
    try:
        # Fetch the schema which contains the list of collections
        schema = client.schema.get()
        classes = schema.get('classes', [])
        
        if not classes:
            logger.info("No collections found.")
            return
        
        logger.info("Listing all collections:")
        for idx, collection in enumerate(classes):
            logger.info(f"Collection {idx+1}: {collection['class']}")
    
    except Exception as e:
        logger.error(f"Error fetching collections: {e}")

def get_record_count(index_name):
    """Get the record count for a specific collection (index)."""
    client = get_weaviate_client()

    try:
        
        logger.info(f"Start Raw query")
        # Query for the count of records in the specific index (class)
        result = client.query.get(index_name, ["title"]).do()

        # Log the raw result to check its structure
        logger.info(f"Raw query result: {result}")

        # Check if the expected data is present in the result
        if "data" in result and "Get" in result["data"] and index_name in result["data"]["Get"]:
            record_count = result["data"]["Get"][index_name]
            logger.info(f"Record count for index '{index_name}': {len(record_count)}")
            return len(record_count)
        else:
            logger.warning(f"No records found or invalid structure for index '{index_name}'.")
            return 0

    except Exception as e:
        logger.error(f"Error fetching record count for {index_name}: {e}")
        return None
def delete_index(index_name):
    """Delete an index (class) in Weaviate."""
    client = get_weaviate_client()

    try:
        # Check if the class (index) exists
        schema = client.schema.get()
        class_names = [cls["class"] for cls in schema["classes"]]

        if index_name in class_names:
            # Delete the class (index)
            client.schema.delete_class(index_name)
            logger.info(f"Index '{index_name}' has been deleted.")
        else:
            logger.warning(f"Index '{index_name}' does not exist.")

    except Exception as e:
        logger.error(f"Error deleting index '{index_name}': {e}")





def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True})
    return embedding_model


def search_weaviate(query):
    client = get_weaviate_client()
    embedding_model = get_embedding_model()
    vector = embedding_model.embed_query(query)
    #WebScrapedData
    results = client.query.get("WebScrapedData", ["title", "content", "url"]) \
        .with_near_vector({"vector": vector}) \
        .with_limit(5) \
        .do()
    return results

def rag_query(query):
    # Initialize the LLM model
    API_URL = "https://granite-8b-code-instruct-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443"
    API_KEY = "bec17dd49be91a48f07b595f8065c3d5"
    llm = VLLMOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=API_URL + "/v1",
        model_name="granite-8b-code-instruct-128k",
        model_kwargs={"stop": ["."]},
    )
    # Search Weaviate for relevant documents
    search_results = search_weaviate(query)

    # Combine the retrieved content with the query for LLM input
    context = ""
    for result in search_results["data"]["Get"]["WebScrapedData"]:
        context += f"Title: {result['title']}\nContent: {result['content']}\n\n"

    # Send the query and context to the LLM model for response
    input_text = f"Query: {query}\n\nContext:\n{context}"
    llm_response = llm.invoke(input_text)
    return llm_response


def get_sample_records(index_name, limit=10):
    """Fetch sample records from the specified index."""
    query = f"""
    {{
        Get {{
            {index_name} (
                limit: {limit}
            ) {{
                page_content
                metadata
            }}
        }}
    }}
    """
    try:
        weaviate_client = get_weaviate_client()
        response = weaviate_client.query.raw(query)
        records = response.get("data", {}).get("Get", {}).get(index_name, [])
        if not records:
            logger.info(f"No records found in {index_name}.")
        else:
            logger.info(f"Fetched {len(records)} records from {index_name}.")
            for record in records:
                logger.info(f"Record: {record}")
        return records
    except Exception as e:
        logger.error(f"Error fetching records from index {index_name}: {e}")
        return []

if __name__ == "__main__":

    #list_collections()

    index_name = "WebScrapedData"  # Replace with your Weaviate index name
    
    records = get_sample_records(index_name="WebScrapedData", limit=10)
    if records:
        print("Sample records:")
        for record in records:
            print(record)
    else:
        print("No records found in the index.")

    # top_records = get_top_records(index_name)
    
    # if top_records:
    #     for i, record in enumerate(top_records, 1):
    #         print(f"Record {i}:\nContent: {record['page_content']}\nMetadata: {record['metadata']}\n")
    # else:
    #     print(f"No records found for index: {index_name}")

    #fetch_top_10_records_and_count(index_name)
    
    # record_count = get_record_count(index_name)
    # if record_count is not None:
    #     logger.info(f"Total records in '{index_name}': {record_count}")
    
    #delte index
    #delete_index(index_name)

    #llm query
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # query = "What's the difference between OpenShift and OKD?"
    # search_results = rag_query(query)
    # print(search_results)