from bs4 import BeautifulSoup
from loguru import logger
from langchain_community.document_loaders.web_base import WebBaseLoader


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

    logger.info(f"Documentation pages found: {pages}")
    return pages


if __name__ == "__main__":
    pages = get_product_pages(
        "red_hat_openshift_ai_self-managed",
        "2.16",
        "en",
    )
