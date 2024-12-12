import os
from typing import List, NamedTuple

import kfp
from kfp import dsl, kubernetes
from kfp.dsl import Artifact, Input, Output


# Function to scrape and convert website content to Markdown
@dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        "beautifulsoup4==4.12.2",
        "html2text==2024.2.26",
        "langchain==0.1.12",
        "lxml==5.1.0",
        "pypdf==4.0.2",
        "tqdm==4.66.2",
        "weaviate-client==3.26.2",
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
    base_image="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/pytorch:2024.1",
    packages_to_install=[
        "langchain==0.1.12",
        "weaviate-client==3.26.2",
        "sentence-transformers==2.4.0",
        "einops==0.7.0",
        "html2text==2024.2.26",
    ],
)
def process_and_store(input_artifact: Input[Artifact], url: str):
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
             # Add the URL to the metadata
            split.metadata["source"] = url
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
    with open(input_artifact.path) as input_file:
        html_artifact = input_file.read()

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


@dsl.pipeline(name="Document Ingestion Pipeline")
def website_ingestion_pipeline(url: str):
    #url = "https://www.redhat.com/en/topics/containers/red-hat-openshift-okd"
    scrape_website_task=scrape_website(url=url)
    process_and_store_task=process_and_store(url=url, input_artifact=scrape_website_task.outputs["html_artifact"])

    process_and_store_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")
    kubernetes.add_toleration(process_and_store_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")

    kubernetes.use_secret_as_env(
        process_and_store_task,
        secret_name="weaviate-api-key-secret", # noqa: S106
        secret_key_to_env={"AUTHENTICATION_APIKEY_ALLOWED_KEYS": "WEAVIATE_API_KEY"},
    )
    process_and_store_task.set_env_variable("WEAVIATE_HOST", "http://weaviate-vector-db")
    process_and_store_task.set_env_variable("WEAVIATE_PORT", "8080")    

if __name__ == "__main__":
    KUBEFLOW_ENDPOINT = os.getenv("KUBEFLOW_ENDPOINT")
    WEBSITE_URL = os.getenv("WEBSITE_URL")
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
        "url": WEBSITE_URL
        }
    )

    # compiler.Compiler().compile(ingestion_pipeline, 'pipeline.yaml')