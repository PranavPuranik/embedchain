from langchain_community.vectorstores.azuresearch import AzureSearch
from embedchain.config.vector_db.base import BaseVectorDbConfig
from embedchain.embedder.base import BaseEmbedder
from .base import BaseVectorDB

class AzureSearchVectorDB(BaseVectorDB):
    """AzureSearch implementation of BaseVectorDB."""

    def __init__(self, config: BaseVectorDbConfig):
        """Initialize the AzureSearchVectorDB with the given config."""
        super().__init__(config)
        self.client = self._get_or_create_db()

    def _initialize(self):
        """Initialize the AzureSearch client and set up collections."""
        self._get_or_create_collection()

    def _get_or_create_db(self):
        """Set up the AzureSearch client."""
        return AzureSearch(
            endpoint=self.config.azure_endpoint,
            api_key=self.config.openai_api_key,
            index_name=self.config.index_name
        )

    def _get_or_create_collection(self):
        """Ensure the collection (index) is created."""
        # Assuming `self.client` has a method to create or fetch collections (indexes).
        if not self.client.index_exists():
            self.client.create_index()

    def add(self, embeddings, metadata=None):
        """Add vectors and metadata to the AzureSearch database."""
        self.client.add_vectors(vectors=embeddings, metadata=metadata)

    def query(self, vector, top_k=10):
        """Query the AzureSearch database for similar vectors."""
        results = self.client.search_vector(vector, top_k=top_k)
        return results

    def get(self, ids):
        """Retrieve vectors from the database by their IDs."""
        return self.client.get_vectors_by_ids(ids)

    def count(self) -> int:
        """Count the number of vectors in the database."""
        return self.client.get_vector_count()

    def reset(self):
        """Reset the AzureSearch index."""
        self.client.delete_index()
        self._get_or_create_collection()

    def set_collection_name(self, name: str):
        """Set the collection (index) name."""
        self.config.index_name = name
        self._get_or_create_collection()

    def delete(self, ids):
        """Delete vectors from the database by their IDs."""
        self.client.delete_vectors(ids)
