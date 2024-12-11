import weaviate
import os
from langchain_community.llms import VLLMOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings



# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to get Weaviate client
def get_weaviate_client():
    """Get the Weaviate client."""
    logger.info("Fetching Weaviate client...")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    weaviate_host = os.getenv("WEAVIATE_HOST")
    weaviate_port = os.getenv("WEAVIATE_PORT")

    if not all([weaviate_api_key, weaviate_host, weaviate_port]):
        logger.error("Weaviate config not present. Check host, port, and API key.")
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
    API_URL = os.getenv("API_URL")
    API_KEY = os.getenv("API_KEY")

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

# Function to scrape and convert website content to Markdown
def scrape_website(url: str):
    import requests
    import logging        
    from bs4 import BeautifulSoup
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    """Scrape and convert the website's HTML to Markdown, split into chunks."""
    logger.info(f"Starting scraping and conversion for URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching the URL {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    main_content = soup.find("body")  # Adjust if necessary to locate main content
    
    if not main_content:
        logger.error("Could not find main content on the webpage.")
        return []

    # html_content = str(main_content)
    # # Writing splits to file to be passed to next step
    # with open(html_artifact.path, "w") as f:
    #     f.write(html_content)
    return str(main_content)

def process_and_store(html_artifact: str, url: str):
    import weaviate
    import os
    import logging
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain_community.document_transformers import Html2TextTransformer
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Weaviate

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

   # Function to get Weaviate client
    def get_weaviate_client():
        """Get the Weaviate client."""
        logger.info("Fetching Weaviate client...")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        weaviate_host = os.getenv("WEAVIATE_HOST")
        weaviate_port = os.getenv("WEAVIATE_PORT")

        if not all([weaviate_api_key, weaviate_host, weaviate_port]):
            logger.error("Weaviate config not present. Check host, port, and API key.")
            exit(1)

        auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key)
        return weaviate.Client(
            url=f"{weaviate_host}:{weaviate_port}",
            auth_client_secret=auth_config,
        )

    def create_index(weaviate_client, index_name, properties):
        """
        Create an index (class) in Weaviate.

        Args:
            weaviate_client (weaviate.Client): The Weaviate client instance.
            index_name (str): The name of the index (class) to create.
            properties (list): List of properties for the class schema. Each property is a dictionary with keys `name` and `dataType`.

        Example:
            properties = [
                {"name": "page_content", "dataType": ["text"]},
                {"name": "metadata", "dataType": ["text"]},
            ]
        """
        try:
            # Check if the index already exists
            existing_classes = [cls["class"] for cls in weaviate_client.schema.get()["classes"]]
            if index_name in existing_classes:
                logger.info(f"Index '{index_name}' already exists. Skipping creation.")
                return

            # Define the schema for the new index
            schema = {
                "class": index_name,
                "properties": properties,
            }

            # Create the index
            weaviate_client.schema.create_class(schema)
            logger.info(f"Successfully created index '{index_name}'.")

        except Exception as e:
            logger.error(f"Error creating index '{index_name}': {e}")

    def convert_to_md(html_content, url):
        # Convert HTML to Markdown using Html2Text
        transformer = Html2TextTransformer()
        document = Document(page_content=html_content, metadata={"url": url})
        md_document = transformer.transform_documents([document])[0]

        # Split Markdown into sections based on headers
        headers_to_split_on = [
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
            ("####", "Header4"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=True)
        header_splits = markdown_splitter.split_text(md_document.page_content)

        # Further split content into chunks for Weaviate ingestion
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=256)
        splits = text_splitter.split_documents(header_splits)

        # Prepare for JSON format with content and metadata
        json_splits = []
        for split in splits:
            content_header = f"Section: {split.metadata.get('Header1', '')}"
            for header_name in ["Header2", "Header3", "Header4"]:
                if header_name in split.metadata:
                    content_header += f" / {split.metadata[header_name]}"
            content_header += "\n\nContent:\n"
            split.page_content = content_header + split.page_content
            json_splits.append({"page_content": split.page_content, "metadata": split.metadata})

        logger.info(f"Successfully processed and converted content for {url}")
        return json_splits

    # Function to ingest data to Weaviate
    def ingest(index_name, splits, weaviate_client):
        """Ingest documents into Weaviate."""
        logger.info(f"Starting ingestion for index: {index_name}")
        try:
            #model_kwargs = {"trust_remote_code": True, "device": "cuda"}
            model_kwargs = {"trust_remote_code": True}
            embeddings = HuggingFaceEmbeddings(
                model_name="nomic-ai/nomic-embed-text-v1",
                model_kwargs=model_kwargs,
                show_progress=True,
            )

            db = Weaviate(
                embedding=embeddings,
                client=weaviate_client,
                index_name=index_name,
                text_key="page_content",
            )
            db.add_documents(splits)
            logger.info(f"Successfully uploaded documents to {index_name}")
        except Exception as e:
            logger.error(f"Error during ingestion for index {index_name}: {e}")


    """Process the website content and add it to the Weaviate store."""
    logger.info(f"Starting processing for URL: {url}")
    weaviate_client = get_weaviate_client()


    # Ensure the index (class) exists
    index_name = "WebScrapedData"
    properties = [
        {"name": "page_content", "dataType": ["text"]},
        {"name": "metadata", "dataType": ["text"]},
    ]
    create_index(weaviate_client, index_name, properties)

    # Reading artifact from previous step into variable
    # with open(input_artifact.path) as input_file:
    #     html_artifact = input_file.read()

    logger.info(f"html_artifact: {html_artifact}")

    # Scrape and process the website
    scraped_data = convert_to_md(html_artifact, url)  

    if not scraped_data:
        logger.warning(f"No data found for {url}. Skipping ingestion.")
        return

    # Prepare the data for ingestion
    document_splits = [("WebScrapedData", [Document(page_content=split["page_content"], metadata=split["metadata"]) for split in scraped_data])]

    # Ingest data in batches to Weaviate
    for index_name, splits in document_splits:
        ingest(index_name=index_name, splits=splits, weaviate_client=weaviate_client)

    logger.info(f"Finished processing for URL: {url}")



def website_ingestion_pipeline():
    url = "https://www.redhat.com/en/topics/containers/red-hat-openshift-okd"
    scrape_website_task=scrape_website(url=url)
    process_and_store(url=url, html_artifact=scrape_website_task)


if __name__ == "__main__":

    #list_collections()

    index_name = "WebScrapedData"  # Replace with your Weaviate index name
    
    #website_ingestion_pipeline()
    
    #get sample records    
    # records = get_sample_records(index_name="WebScrapedData", limit=10)
    # if records:
    #     print("Sample records:")
    #     for record in records:
    #         print(record)
    # else:
    #     print("No records found in the index.")

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