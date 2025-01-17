import os

from elasticsearch import Elasticsearch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import ElasticsearchStore
from langchain_core.documents import Document


class ElasticSearchDocumentLoader:
    def __init__(
        self,
        documents: list[Document],
        collection_name: str,
        host: str,
        port: str | None = "9200",
        protocol: str = "http",
        verify_certs: bool = False,
        elastic_user: str = os.environ("ELASTIC_USER"),
        elastic_password: str = os.environ("ELASTIC_PASSWORD"),
        embedding_model: str = "nomic-ai/nomic-embed-text-v1",
    ):
        self.documents = list(documents)
        self.collection_name = collection_name
        if port:
            self.connection_url = f"{protocol}://{host}:port"
        else:
            self.connection_url = f"{protocol}://{host}"
        self.verify_certs = verify_certs
        self.elastic_user = elastic_user
        self.elastic_password = elastic_password

        self.connection = self._create_connection

        self._test_connection()

        self.embedding_model = embedding_model
        self.embedding = self._create_embedding()

        self.store = self._create_store()

    def _create_connection(self) -> Elasticsearch:
        connection = Elasticsearch(
            self.connection_url, basic_auth=(self.elastic_user, self.elastic_password), verify_certs=self.verify_certs
        )

        return connection

    def _create_embedding(self) -> HuggingFaceEmbeddings:
        model_kwargs = {"trust_remote_code": True, "device": "cuda"}

        embeddings = HuggingFaceEmbeddings(
            model_name=self.em,
            model_kwargs=model_kwargs,
            show_progress=True,
        )

        return embeddings

    def _create_store(self) -> ElasticsearchStore:
        store = ElasticsearchStore(
            embedding=self.embedding, index_name=self.collection_name, es_connection=self.connection
        )
        return store

    def _test_connection(self):
        self.connection.info()

    def load_documents(self):
        self.store.add_documents(self.documents)
