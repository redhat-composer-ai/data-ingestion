from data_ingestion.redhat_products.redhat_product_documentation import RedHatProductDocumentation


def get_documents_list(product: str, version: str, language: str = "en") -> list[str]:
    rh_product = RedHatProductDocumentation(product=product, version=version, language=language)
    product_doc_pages = rh_product.product_pages()
    return product_doc_pages
