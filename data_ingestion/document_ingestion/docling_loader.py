"""
A website base loader modeled after langchain WebBaseLoader
https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_loaders/web_base.py
"""

from collections.abc import Iterator
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from loguru import logger
from slugify import slugify


class DoclingWebsiteBaseLoader(BaseLoader):
    def __init__(
        self,
        web_paths: str | list[str],
        save_as_markdown: bool = True,
        save_target_path: Path | str = "./data/raw",
    ):
        # make web_paths a list if the user only provided a single value
        self.web_paths = list(web_paths)
        self._converter = DocumentConverter()
        self.save_as_markdown = save_as_markdown
        self.requests_timeout = 10
        if self.save_as_markdown:
            self.save_target_path = Path(save_target_path)
            # create the parent folder if it doesn't already exist
            self.save_target_path.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        for url in self.web_paths:
            logger.info(f"Loading document from: {url}")
            docling_doc, soup = self._scrape(url)

            page_contents = docling_doc.document.export_to_markdown()
            metadata = self._build_metadata(soup, url)

            if self.save_as_markdown:
                self._save_as_markdown(metadata["title"], page_contents)

            yield Document(page_content=page_contents, metadata=metadata)

    def _scrape(self, url: str) -> tuple[DoclingDocument, BeautifulSoup]:
        docling_doc = self._scrape_docling(url)
        soup = self._scrape_soup(url)
        return docling_doc, soup

    def _scrape_docling(self, url: str) -> DoclingDocument:
        converter = self._converter
        docling_doc = converter.convert(url)

        return docling_doc

    def _scrape_soup(self, url: str) -> BeautifulSoup:
        html_text = requests.get(url, timeout=self.requests_timeout).text
        soup = BeautifulSoup(html_text)
        return soup

    def _build_metadata(self, soup: BeautifulSoup, url: str) -> dict:
        """Build metadata from BeautifulSoup output."""
        metadata = {"source": url}
        if title := soup.find("title"):
            metadata["title"] = title.get_text()
        if description := soup.find("meta", attrs={"name": "description"}):
            metadata["description"] = description.get("content", "No description found.")
        if html := soup.find("html"):
            metadata["language"] = html.get("lang", "No language found.")
        logger.debug(f"metadata: {metadata}")

        return metadata

    def _save_as_markdown(self, title: str, markdown_text: str):
        target_filename = f"{slugify(title)}.md"
        target_filepath = self.save_target_path / target_filename
        logger.info(f"Saving document contents to {target_filepath}")
        with open(target_filepath, "w") as markdown_file:
            markdown_file.write(markdown_text)
