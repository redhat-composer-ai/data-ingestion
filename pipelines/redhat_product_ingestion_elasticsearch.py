import os
import sys

from dotenv import load_dotenv
from kfp_helper import execute_pipeline_run

import kfp.compiler
from kfp import dsl
from kfp.dsl import Artifact

# hacky way to add the project root to be able to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_ingestion.redhat_products.ingestion import get_product_documents

@kfp.dsl.pipeline()
def ingestion_pipeline(product: str = "red_hat_openshift_ai_self-managed", version: str = "2.16", language: str= "en"):
    get_documents_list_task = get_product_documents(product=product, version=version, language=language)
    # convert_to_markdown_task = convert_to_markdown(raw_document=get_documents_list_task.output)
    # ingest_document(document=convert_to_markdown_task.output)


if __name__ == "__main__":
    arguments = {"product": "red_hat_openshift_ai_self-managed", "version": "2.16","language": "en"}
    execute_pipeline_run(pipeline=ingestion_pipeline, experiment="data-ingestion", arguments=arguments)
    # compile_pipeline(pipeline=ingestion_pipeline)
