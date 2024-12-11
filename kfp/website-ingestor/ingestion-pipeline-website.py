import os
import logging
import requests
from typing import List, NamedTuple
import weaviate
from kfp import dsl
from kfp.dsl import Input, Output, Artifact
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dsl.component
def get_weaviate_client() -> weaviate.Client:
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

@dsl.component
def scrape_and_convert_to_md(url: str) -> List[dict]:
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

    html_content = str(main_content)

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

@dsl.component
def create_index(weaviate_client: weaviate.Client, index_name: str, properties: List[dict]):
    """Create an index (class) in Weaviate."""
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

@dsl.component
def ingest(index_name: str, splits: List[Document], weaviate_client: weaviate.Client):
    """Ingest documents into Weaviate."""
    logger.info(f"Starting ingestion for index: {index_name}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1",
            model_kwargs={"trust_remote_code": True},
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

@dsl.pipeline(name="Document Ingestion Pipeline")
def document_ingestion_pipeline(url: str):
    weaviate_client_task = get_weaviate_client()
    
    index_name = "WebScrapedData"
    properties = [
        {"name": "page_content", "dataType": ["text"]},
        {"name": "metadata", "dataType": ["text"]},
    ]
    create_index_task = create_index(
        weaviate_client=weaviate_client_task.output,
        index_name=index_name,
        properties=properties,
    )

    scrape_task = scrape_and_convert_to_md(url=url)
    document_splits = [
        Document(page_content=split["page_content"], metadata=split["metadata"]) 
        for split in scrape_task.output
    ]
    ingest_task = ingest(
        index_name=index_name,
        splits=document_splits,
        weaviate_client=weaviate_client_task.output
    )

    # Set tolerations, GPU limits, and secrets if necessary
    # Example:
    # kubernetes.use_secret_as_env(ingest_task, secret_name="weaviate-api-key-secret", secret_key_to_env={"WEAVIATE_API_KEY": "WEAVIATE_API_KEY"})
    # ingest_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")

if __name__ == "__main__":
    from kfp import Client
    KUBEFLOW_ENDPOINT = os.getenv("KUBEFLOW_ENDPOINT")
    client = Client(host=KUBEFLOW_ENDPOINT)
    
    # Run the pipeline
    result = client.create_run_from_pipeline_func(
        document_ingestion_pipeline,
        arguments={"url": "https://example.com"},
        experiment_name="web-scraping-experiment"
    )
