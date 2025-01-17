from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentSplitter:
    def __init__(self, documents: list[Document]):
        self.documents = list(documents)
        self.chunk_size = 1024
        self.chunk_overlap = 40
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def splits(self) -> list[Document]:
        document_splits = self.splitter.split_documents(self.documents)
        return document_splits
