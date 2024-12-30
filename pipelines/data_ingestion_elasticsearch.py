import os

from dotenv import load_dotenv
import kfp.compiler

from kfp import dsl
from kfp.dsl import Artifact, Input, Output

from kfp_helper import execute_pipeline_run, compile_pipeline

load_dotenv(override=True)

kubeflow_endpoint = os.environ["KUBEFLOW_ENDPOINT"]
base_image = os.getenv("BASE_IMAGE", "image-registry.openshift-image-registry.svc:5000/openshift/python:latest")


@dsl.component(base_image=base_image)
def download_document() -> Artifact:
    document = ""
    return document


@dsl.component(base_image=base_image)
def convert_to_markdown(raw_document: Artifact) -> Artifact:
    with open(raw_document.path) as f:
        document = f.readlines()

    markdown_document = document

    return markdown_document


@dsl.component(base_image=base_image)
def ingest_document(document: Artifact):
    with open(document.path) as f:
        document = f.readlines()


@kfp.dsl.pipeline()
def ingestion_pipeline():
    download_document_task = download_document()
    convert_to_markdown_task = convert_to_markdown(raw_document=download_document_task.output)
    ingest_document_task = ingest_document(document=convert_to_markdown_task.output)


if __name__ == "__main__":
    # execute_pipeline_run(pipeline=ingestion_pipeline, experiment="data-ingestion")
    compile_pipeline(pipeline=ingestion_pipeline)
