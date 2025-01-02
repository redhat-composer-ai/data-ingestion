import os
from typing import List, NamedTuple

import kfp
from kfp import dsl, kubernetes
from kfp.dsl import Artifact, Input, Output


# Function to scrape and convert website content to Markdown
@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "beautifulsoup4==4.12.2",
        "requests==2.32.3",
        "html2text==2024.2.26",
        "lxml==5.1.0",
        "pypdf==4.0.2",
        "tqdm==4.66.2",
        "torch==2.4.0",
    ],
)
def scrape_website(url: str, html_artifact: Output[Artifact]):
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

    html_content = str(main_content)
    # Writing splits to file to be passed to next step
    with open(html_artifact.path, "w") as f:
        f.write(html_content)


@dsl.component(
    base_image="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/minimal-gpu:2024.2",
    packages_to_install=[
        "langchain-community==0.3.8",
        "langchain==0.3.8",
        "sentence-transformers==2.4.0",
        "einops==0.7.0",
        "html2text==2024.2.26",
        "elastic-transport==8.15.1",
        "elasticsearch==8.16.0",
        "langchain-elasticsearch==0.3.0",        
    ],
)
def process_and_store(input_artifact: Input[Artifact], url: str, index_name: str):
    from elasticsearch import Elasticsearch
    import os
    import logging
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain_community.document_transformers import Html2TextTransformer
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    from langchain_elasticsearch import ElasticsearchStore

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

   # Function to get Weaviate client
    def get_es_client():
        """Get the Weaviate client."""
        logger.info("Fetching es client...")

        es_user = os.environ.get("ES_USER")
        es_pass = os.environ.get("ES_PASS")
        es_host = os.environ.get("ES_HOST")

        if not es_user or not es_pass or not es_host:
            print("Elasticsearch config not present. Check host, port and api_key")
            exit(1)
        # Iniatilize Elastic client
        es_client = Elasticsearch(es_host, 
                                basic_auth=(es_user, es_pass), 
                                request_timeout=30, 
                                verify_certs=False)

        # # Health check for elastic client connection
        print(f"Elastic Client status: {es_client.health_report()}")

        return es_client
    
    def create_index(es_client: Elasticsearch, index_name: str, mappings: dict = None):
        """
        Create an index in Elasticsearch.

        Args:
            es_client (Elasticsearch): The Elasticsearch client instance.
            index_name (str): The name of the index to create.
            mappings (dict, optional): The mappings and settings for the index. Defaults to a predefined structure.

        Returns:
            bool: True if the index was created, False if it already exists.
        """
        try:
            # Check if the index already exists
            if es_client.indices.exists(index=index_name):
                logger.info(f"Index '{index_name}' already exists. Skipping creation.")
                return False

            # Default mappings if none are provided
            if mappings is None:
                mappings = {
                    "mappings": {
                        "properties": {
                            "page_content": {"type": "text"},
                            "metadata": {"type": "text"}
                        }
                    }
                }

            # Create the index
            es_client.indices.create(index=index_name, body=mappings)
            logger.info(f"Successfully created index '{index_name}'.")
            return True

        except Exception as e:
            logger.error(f"Error creating index '{index_name}': {e}")
            raise

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
             # Add the URL to the metadata
            split.metadata["source"] = url
            json_splits.append({"page_content": split.page_content, "metadata": split.metadata})

        logger.info(f"Successfully processed and converted content for {url}")
        return json_splits

    # Function to ingest data to Weaviate
    def ingest(index_name, splits, es_client):
        """Ingest documents into Elasticsearch."""
        logger.info(f"Starting ingestion for index: {index_name}")
        try:
            #model_kwargs = {"trust_remote_code": True, "device": "cuda"}
            model_kwargs = {"trust_remote_code": True}
            embeddings = HuggingFaceEmbeddings(
                model_name="nomic-ai/nomic-embed-text-v1",
                model_kwargs=model_kwargs,
                show_progress=True,
            )

            db = ElasticsearchStore(
                index_name=index_name.lower(),  # index names in elastic must be lowercase
                embedding=embeddings,
                es_connection=es_client,
            )
            db.add_documents(splits)
            logger.info(f"Successfully uploaded documents to {index_name}")
        except Exception as e:
            logger.error(f"Error during ingestion for index {index_name}: {e}")


    """Process the website content and add it to the Weaviate store."""
    logger.info(f"Starting processing for URL: {url}")
    es_client = get_es_client()


    # Ensure the index (class) exists
    create_index(es_client, index_name)

    # Reading artifact from previous step into variable
    with open(input_artifact.path) as input_file:
        html_artifact = input_file.read()

    # Scrape and process the website
    scraped_data = convert_to_md(html_artifact, url)  

    if not scraped_data:
        logger.warning(f"No data found for {url}. Skipping ingestion.")
        return

    # Prepare the data for ingestion
    document_splits = [(index_name, [Document(page_content=split["page_content"], metadata=split["metadata"]) for split in scraped_data])]

    # Ingest data in batches to Weaviate
    for index_name, splits in document_splits:
        ingest(index_name=index_name, splits=splits, es_client=es_client)

    logger.info(f"Finished processing for URL: {url}")


@dsl.pipeline(name="Document Ingestion Pipeline")
def website_ingestion_pipeline(url: str, index_name: str):
    #url = "https://www.redhat.com/en/topics/containers/red-hat-openshift-okd"
    scrape_website_task=scrape_website(url=url)
    process_and_store_task=process_and_store(url=url,index_name=index_name, input_artifact=scrape_website_task.outputs["html_artifact"])

    process_and_store_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")
    kubernetes.add_toleration(process_and_store_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")

    kubernetes.use_secret_as_env(
        process_and_store_task,
        secret_name="elasticsearch-es-elastic-user",
        secret_key_to_env={"elastic": "ES_PASS"},
    )
    process_and_store_task.set_env_variable("ES_HOST", "http://elasticsearch-es-http:9200")
    process_and_store_task.set_env_variable("ES_USER", "elastic")

if __name__ == "__main__":
    KUBEFLOW_ENDPOINT = os.getenv("KUBEFLOW_ENDPOINT")
    WEBSITE_URL = os.getenv("WEBSITE_URL")
    VECTORDB_INDEX = os.getenv("VECTORDB_INDEX").lower()
    print(f"Connecting to kfp: {KUBEFLOW_ENDPOINT}")
    sa_token_path = "/run/secrets/kubernetes.io/serviceaccount/token" # noqa: S105
    if os.path.isfile(sa_token_path):
        with open(sa_token_path) as f:
            BEARER_TOKEN = f.read().rstrip()
    else:
        BEARER_TOKEN = os.getenv("BEARER_TOKEN")

    sa_ca_cert = "/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"
    if os.path.isfile(sa_ca_cert):
        ssl_ca_cert = sa_ca_cert
    else:
        ssl_ca_cert = None

    client = kfp.Client(
        host=KUBEFLOW_ENDPOINT,
        existing_token=BEARER_TOKEN,
        ssl_ca_cert=None,
    )
    result = client.create_run_from_pipeline_func(
        website_ingestion_pipeline,
        experiment_name="website_ingestion",
        enable_caching=False,
        arguments={
        "url": WEBSITE_URL,
        "index_name": VECTORDB_INDEX
        }
    )

    # compiler.Compiler().compile(ingestion_pipeline, 'pipeline.yaml')