from unittest.mock import patch

import pytest
from bs4 import BeautifulSoup

# Assuming the function is in a module called 'doc_scraper'
from data_ingestion.data.redhat_product_documentation import get_product_pages


@pytest.fixture
def mock_loader():
    """Fixture to mock the WebBaseLoader and its scrape method."""
    with patch("data_ingestion.data.redhat_product_documentation.WebBaseLoader") as mock_loader:
        mock_instance = mock_loader.return_value
        yield mock_instance


@pytest.fixture
def example_en_html():
    """Fixture for example HTML content to mock the scraped webpage."""
    return """
    <html>
        <body>
            <h3 slot="headline"><a href="/en/documentation/html/some-page-1">Page 1</a></h3>
            <h3 slot="headline"><a href="/en/documentation/html/some-page-2">Page 2</a></h3>
            <h3 slot="headline"><a href="/en/other-section/some-page-3">Page 3</a></h3>
        </body>
    </html>
    """


@pytest.fixture
def example_fr_html():
    """Fixture for example HTML content to mock the scraped webpage."""
    return """
    <html>
        <body>
            <h3 slot="headline"><a href="/fr/documentation/html/some-page-1">Page 1</a></h3>
            <h3 slot="headline"><a href="/fr/documentation/html/some-page-2">Page 2</a></h3>
            <h3 slot="headline"><a href="/fr/other-section/some-page-3">Page 3</a></h3>
        </body>
    </html>
    """


def test_get_product_pages_success(mock_loader, example_en_html):
    """Test successful retrieval and filtering of product pages."""
    # Mock the scrape method to return the example HTML content
    mock_loader.scrape.return_value = BeautifulSoup(example_en_html, "lxml")

    # Call the function with test parameters
    product = "test-product"
    version = "1.0"
    language = "en"

    pages = get_product_pages(product, version, language)

    # Assert the result
    assert pages == ["/en/documentation/html-single/some-page-1", "/en/documentation/html-single/some-page-2"]


def test_get_product_non_english_pages_success(mock_loader, example_fr_html):
    """Test successful retrieval and filtering of product pages."""
    # Mock the scrape method to return the example HTML content
    mock_loader.scrape.return_value = BeautifulSoup(example_fr_html, "lxml")

    # Call the function with test parameters
    product = "test-product"
    version = "1.0"
    language = "fr"

    pages = get_product_pages(product, version, language)

    # Assert the result
    assert pages == ["/fr/documentation/html-single/some-page-1", "/fr/documentation/html-single/some-page-2"]


def test_get_product_pages_no_links(mock_loader):
    """Test the case where no links are found."""
    # Mock the scrape method to return empty content
    mock_loader.scrape.return_value = BeautifulSoup("<html></html>", "lxml")

    # Call the function with test parameters
    product = "test-product"
    version = "1.0"
    language = "en"

    pages = get_product_pages(product, version, language)

    # Assert the result
    assert pages == []


def test_get_product_pages_invalid_language(mock_loader):
    """Test the case where links do not match the language prefix."""
    html_content = """
    <html>
        <body>
            <h3 slot="headline"><a href="/fr/documentation/html/some-page-1">Page 1</a></h3>
        </body>
    </html>
    """
    # Mock the scrape method to return the custom HTML content
    mock_loader.scrape.return_value = BeautifulSoup(html_content, "lxml")

    # Call the function with test parameters
    product = "test-product"
    version = "1.0"
    language = "en"

    pages = get_product_pages(product, version, language)

    # Assert the result
    assert pages == []
