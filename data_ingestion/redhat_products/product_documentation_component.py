import os

from kfp import dsl

image = os.getenv("COMPONENT_IMAGE", "quay.io/redhat-composer-ai/data-ingestion")
tag = os.getenv("COMPONENT_IMAGE_TAG", "latest")


@dsl.component(target_image=f"{image}:{tag}")
def get_documents_list_component(product: str, version: str, language: str) -> list:
    from data_ingestion.redhat_products.product_documentation import get_documents_list

    documents_list = get_documents_list(product=product, version=version, language=language)

    return documents_list
