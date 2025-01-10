from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter
from langchain_community.document_loaders.web_base import WebBaseLoader
from loguru import logger


class RedHatProductDocumentation:
    def __init__(self, product: str, version: str, language: str = "en"):
        self.documentation_url = "https://access.redhat.com/documentation"
        self.product = product
        self.version = str(version)
        self.language = language
        self.product_url = [self.documentation_url + "/" + self.language + "/" + self.product + "/" + self.version]

    def product_pages(self) -> list[str]:
        """Get the list of pages from the Red Hat product documentation."""
        soup = self._scrape()

        # Select only the element titles that contain the links to the documentation pages
        filtered_elements = soup.find_all("h3", attrs={"slot": "headline"})
        new_soup = BeautifulSoup("", "lxml")
        for element in filtered_elements:
            new_soup.append(element)

        # Extract all the links
        links = []
        for match in new_soup.findAll("a"):
            logger.debug(f"Found unfiltered link: {match.get('href')}")
            links.append(match.get("href"))

        # Filter the links to include only those that start with "/en/documentation"
        filtered_links = []
        for url in links:
            if url.startswith(f"/{self.language}/documentation"):
                logger.debug(f"Found filtered link: {url}")
                filtered_links.append(url)

        # Swap the URL with the single page link
        pages = []
        for link in filtered_links:
            if "/html/" in link:
                pages.append(link.replace("/html/", "/html-single/"))

        # add the base URL to all of the found pages
        pages = [f"{self.documentation_url}/{page}" for page in pages]

        logger.info(f"Found {len(pages)} documents")
        logger.debug(f"Documentation pages found: {pages}")
        return pages

    def _scrape(self) -> BeautifulSoup:
        logger.info(f"Scraping {self.product_url} for document list")
        loader = WebBaseLoader(self.product_url)
        soup = loader.scrape()

        return soup


def get_product_pages(product: str, version: str, language: str) -> list[str]:
    """Get the list of pages from the Red Hat product documentation."""

    # Load the Red Hat documentation page
    url = ["https://access.redhat.com/documentation/" + language + "/" + product + "/" + version]
    logger.info(f"Scraping {url} for document list")

    loader = WebBaseLoader(url)
    soup = loader.scrape()

    # Select only the element titles that contain the links to the documentation pages
    filtered_elements = soup.find_all("h3", attrs={"slot": "headline"})
    new_soup = BeautifulSoup("", "lxml")
    for element in filtered_elements:
        new_soup.append(element)

    # Extract all the links
    links = []
    for match in new_soup.findAll("a"):
        logger.debug(f"Found unfiltered link: {match.get('href')}")
        links.append(match.get("href"))

    # Filter the links to include only those that start with "/en/documentation"
    filtered_links = []
    for url in links:
        if url.startswith(f"/{language}/documentation"):
            logger.debug(f"Found filtered link: {url}")
            filtered_links.append(url)

    # links = [url for url in links if url.startswith("/en/documentation")]  # Filter out unwanted links

    # Swap the URL with the single page link
    pages = []
    for link in filtered_links:
        if "/html/" in link:
            pages.append(link.replace("/html/", "/html-single/"))

    logger.info(f"Found {len(pages)} documents")
    logger.debug(f"Documentation pages found: {pages}")
    return pages


def download_product_documentation_pages(
    product_name: str, product_version: str, pages: list[str], target_folder: str = "./data/raw"
) -> Path:
    product_folder = Path(target_folder, product_name)
    versioned_product_folder = product_folder / product_version
    versioned_product_folder.mkdir(parents=True, exist_ok=True)

    for page in pages:
        download_documentation_page(product_name, product_version, page, target_folder=versioned_product_folder)

    return versioned_product_folder


def download_documentation_page(
    product_name: str, product_version: str, page: str, target_folder: str, root_url: str = "https://docs.redhat.com"
) -> Path:
    page_url = urljoin(root_url, page)

    page_markdown = convert_page_to_markdown(page_url=page_url)

    page_filename = f"{Path(page).name}.md"
    page_filepath = target_folder / page_filename

    with open(page_filepath, "w") as markdown_file:
        markdown_file.write(page_markdown)

    return page_filename


def convert_page_to_markdown(page_url: str) -> str:
    converter = DocumentConverter()

    logger.info(f"Converting page to markdown: {page_url}")
    result = converter.convert(page_url)
    results_markdown = result.document.export_to_markdown()

    return str(results_markdown)


if __name__ == "__main__":
    product = "red_hat_openshift_ai_self-managed"
    version = "2.16"
    language = "en"

    documents = RedHatProductDocumentation(product=product, version=version, language=language)
    print(documents.product_pages())
    # pages = get_product_pages(product, version, language)
    # download_product_documentation_pages(product, version, pages)
