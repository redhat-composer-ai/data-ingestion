import json
import os
from typing import List, NamedTuple

from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def load_documents() -> List:
    Product = NamedTuple("Product", product=str, product_full_name=str, version=str, language=str)

    products = [
        Product("red_hat_openshift_ai_self-managed", "Red Hat OpenShift AI Self-Managed", "2.14", "en-US"),
        # Product('openshift_container_platform', 'Red Hat OpenShift Container Platform', '4.17', 'en-US'),
        # Product('red_hat_enterprise_linux', 'Red Hat Enterprise Linux 9', '9', 'en-US'),
        # Product('red_hat_ansible_automation_platform', 'Red Hat Ansible Automation Platform', '2.5', 'en-US'),
    ]
    return products


def format_documents(documents: List, splits_artifact):
    class RedHatDocumentationLoader(WebBaseLoader):
        """Load `Red Hat Documentation` single-html webpages."""

        def load(self) -> List[Document]:
            """Load webpages as Documents."""
            soup = self.scrape()
            title = soup.select_one("h1", {"class": "title"}).text  # Get title

            # Get main content
            book = soup.select_one(".book")
            if book:
                soup = book
            else:
                article = soup.select_one(".article")
                if article:
                    soup = article
                else:
                    soup = None

            if soup is not None:
                # Remove unwanted sections
                unwanted_classes = [
                    "producttitle",
                    "subtitle",
                    "abstract",
                    "legalnotice",
                    "calloutlist",
                    "callout",
                ]
                for unwanted_class in unwanted_classes:
                    for div in soup.find_all("div", {"class": unwanted_class}):
                        div.decompose()
                    for span in soup.find_all("span", {"class": unwanted_class}):
                        span.decompose()
                    for header in soup.find_all("h2", {"class": unwanted_class}):
                        header.decompose()
                for hr in soup.find_all("hr"):
                    hr.decompose()

                # Find and delete anchor tag with content "Legal Notice"
                for anchor in soup.find_all("a"):
                    if anchor.text == "Legal Notice":
                        anchor.decompose()

                # Unwrap unwanted tags
                unwrap_tags = ["div", "span", "strong", "section"]
                for tag in unwrap_tags:
                    for match in soup.findAll(tag):
                        match.unwrap()

                # Transform description titles
                for dt in soup.find_all("dt"):
                    if dt.string:
                        dt.string.replace_with(f"-> {dt.string}")

                # Transform code blocks
                for code in soup.find_all("pre", {"class": "programlisting"}):
                    try:
                        content = code.text
                        code.clear()
                        if "language-yaml" in code["class"]:
                            code.string = f"```yaml\n{content}\n```"
                        elif "language-json" in code["class"]:
                            code.string = f"```json\n{content}\n```"
                        elif "language-bash" in code["class"]:
                            code.string = f"```bash\n{content}\n```"
                        elif "language-python" in code["class"]:
                            code.string = f"```python\n{content}\n```"
                        elif "language-none" in code["class"]:
                            code.string = f"```\n{content}\n```"
                        else:
                            code.string = f"```\n{content}\n```"
                    except Exception as e:
                        print(f"Error processing code block: {e}")
                for code in soup.find_all("pre", {"class": "screen"}):
                    try:
                        content = code.text
                        code.clear()
                        code.string = f"```console\n{content}\n```"
                    except Exception as e:
                        print(f"Error processing code block: {e}")

                # Remove all attributes
                for tag in soup():
                    tag.attrs.clear()

                text = str(soup)  # Convert to string
                text = text.replace("\xa0", " ")  # Replace non-breaking space

            else:
                text = ""

            # Add metadata
            metadata = {"source": self.web_path, "title": title}

            return [Document(page_content=text, metadata=metadata)]

    print("Starting format_documents")

    def get_pages(product, version, language) -> List:
        """Get the list of pages from the Red Hat product documentation."""

        # Load the Red Hat documentation page
        url = ["https://access.redhat.com/documentation/" + language + "/" + product + "/" + version]
        loader = WebBaseLoader(url)
        soup = loader.scrape()
        print(f"URL {url}")
        # Select only the element titles that contain the links to the documentation pages
        filtered_elements = soup.find_all("h3", attrs={"slot": "headline"})
        new_soup = BeautifulSoup("", "lxml")
        for element in filtered_elements:
            new_soup.append(element)
        for match in new_soup.findAll("h3"):
            match.unwrap()

        # Extract all the links
        links = []
        for match in new_soup.findAll("a"):
            links.append(match.get("href"))
        links = [url for url in links if url.startswith("/en/documentation")]  # Filter out unwanted links
        pages = [
            link.replace("/html/", "/html-single/") for link in links if "/html/" in link
        ]  # We want single pages html
        # print(f"{len(links)} links found\n {len(pages)} pages found\n", links, pages)
        return pages

    def split_document(product, version, language, page, product_full_name) -> List:
        """Split a Red Hat documentation page into smaller sections."""

        # Load, parse, and transform to Markdown
        document_url = ["https://docs.redhat.com" + page]
        print(f"Processing: {document_url}")
        loader = RedHatDocumentationLoader(document_url)
        docs = loader.load()
        html2text = Html2TextTransformer()
        md_docs = html2text.transform_documents(docs)

        # Markdown splitter config
        headers_to_split_on = [
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=True)

        # Markdown split
        new_splits: List[Document] = []
        for doc in md_docs:
            md_header_splits = markdown_splitter.split_text(doc.page_content)
            for split in md_header_splits:
                split.metadata |= doc.metadata
                split.metadata["product"] = product
                split.metadata["version"] = version
                split.metadata["language"] = language
                split.metadata["product_full_name"] = product_full_name
            new_splits.extend(md_header_splits)

        # Char-level splitter config
        chunk_size = 2048
        chunk_overlap = 256
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Char-level split
        splits = text_splitter.split_documents(new_splits)
        json_splits = []

        for split in splits:
            content_header = f"Section: {split.metadata['title']}"
            for header_name in ["Header1", "Header2", "Header3"]:
                if header_name in split.metadata:
                    content_header += f" / {split.metadata[header_name]}"
            content_header += "\n\nContent:\n"
            split.page_content = content_header + split.page_content
            json_splits.append({"page_content": split.page_content, "metadata": split.metadata})

        return json_splits

    def generate_splits(product, product_full_name, version, language) -> List:
        """Generate the splits for a Red Hat documentation product."""

        # Find all the pages.
        pages = get_pages(product, version, language)
        print(f"Found {len(pages)} pages:")
        print(pages)

        # Generate the splits.
        print("Generating splits...")
        all_splits = []
        for page in pages:
            splits = split_document(product, version, language, page, product_full_name)
            all_splits.extend(splits)
        print(f"Generated {len(all_splits)} splits.")

        return all_splits

    DocumentSplit = NamedTuple("DocumentSplit", index_name=str, splits=List[str])
    document_splits = []
    for doc in documents:
        product, product_full_name, version, language = doc

        index_name = f"{product}_{language}_{version}".replace("-", "_").replace(".", "_")
        splits = generate_splits(
            product=product, product_full_name=product_full_name, version=version, language=language
        )
        document_splits.append(DocumentSplit(index_name=index_name, splits=splits))

    splits_artifact.append(json.dumps(document_splits))

    return document_splits


def ingest_documents(input_artifact: List):
    # Reading artifact from previous step into variable
    document_splits = []
    for input in input_artifact:
        splits_artifact = input
        document_splits.append(splits_artifact)

    es_user = os.environ.get("ES_USER")
    es_pass = os.environ.get("ES_PASS")
    es_host = os.environ.get("ES_HOST")

    es_client = Elasticsearch(es_host, basic_auth=(es_user, es_pass), request_timeout=30, verify_certs=False)

    # Health check for WEAVIATE_CLIENT connection

    def ingest(index_name, splits):
        # Here we use Nomic AI's Nomic Embed Text model to generate embeddings
        # Adapt to your liking
        model_kwargs = {"trust_remote_code": True, "device": "cpu"}
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

        print(f"Uploading document to collection {index_name}")
        db.add_documents(splits)

    for index_name, splits in document_splits:
        documents = [Document(page_content=split["page_content"], metadata=split["metadata"]) for split in splits]
        ingest(index_name=index_name, splits=documents)

    print("Finished!")


if __name__ == "__main__":
    docs = load_documents()

    splits_artifact = []

    f_docs = format_documents(docs, splits_artifact)
    ingestion = ingest_documents(f_docs)
