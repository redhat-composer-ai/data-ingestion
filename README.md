# Data Ingestion

Repository containing methods used for ingesting data into data sources used by Composer AI.

## Local Environment Setup

1. Install uv if you don't have it already:
    ```
    pip install uv
    ```
1. Install packages using uv:
    ```
    uv sync
    ```
1. Setup pre-commit hooks
    ```
    pre-commit install
    ```
1. Setup your `.env` file by copying the `example.env` to `.env`

## Building an image locally

```
podman build . --platform linux/amd64 -t quay.io/redhat-composer-ai/data-ingestion:<tag>
podman push quay.io/redhat-composer-ai/data-ingestion:<tag>
```

If testing locally, be sure to update your .env file with the correct tag when kicking off the pipeline.

## Running Pipeline

Be sure you have updated your `.env` file

```
uv run python pipelines/redhat_product_ingestion_elasticsearch.py
```
