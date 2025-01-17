from pathlib import Path

from langchain_core.documents import Document

from data_ingestion.document_ingestion.docling_loader import DoclingWebsiteBaseLoader
from data_ingestion.redhat_products.redhat_product_documentation import RedHatProductDocumentation


def get_product_documents(product: str, version: str, language: str = "en") -> list[str]:
    rh_product = RedHatProductDocumentation(product=product, version=version, language=language)
    product_doc_pages = rh_product.product_pages()
    return product_doc_pages


def load_documents(
    product_pages: list[str],
    product: str,
    version: str,
    language: str = "en",
    markdown_root_folder: Path | str = "./data/raw",
) -> tuple[list[Document], Path]:
    target_folder = Path(markdown_root_folder) / product / version / language
    loader = DoclingWebsiteBaseLoader(web_paths=product_pages, save_target_path=target_folder)
    documents = loader.load()
    return documents, target_folder


def main(product: str, version: str, language: str = "en"):
    product_pages = get_product_documents(product=product, version=version, language=language)
    documents = load_documents(product_pages=product_pages, product=product, version=version, language=language) # noqa: F841


if __name__ == "__main__":
    product = "red_hat_openshift_ai_self-managed"
    version = "2.16"
    language = "en"

    main(product=product, version=version, language=language)
    # pages = get_product_pages(product, version, language)
    # download_product_documentation_pages(product, version, pages)
