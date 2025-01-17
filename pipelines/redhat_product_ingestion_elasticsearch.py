import os
import sys

from kfp_helper import execute_pipeline_run

import kfp.compiler

# hacky way to add the project root to be able to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_ingestion.redhat_products.product_documentation_component import get_documents_list_component


@kfp.dsl.pipeline()
def ingestion_pipeline(product: str = "red_hat_openshift_ai_self-managed", version: str = "2.16", language: str = "en"):
    get_documents_list_task = get_documents_list_component(product=product, version=version, language=language) # noqa: F841
    # convert_to_markdown_task = convert_to_markdown(raw_document=get_documents_list_task.output)
    # ingest_document(document=convert_to_markdown_task.output)


if __name__ == "__main__":
    arguments = {"product": "red_hat_openshift_ai_self-managed", "version": "2.16", "language": "en"}
    execute_pipeline_run(pipeline=ingestion_pipeline, experiment="data-ingestion", arguments=arguments)
    # compile_pipeline(pipeline=ingestion_pipeline)
