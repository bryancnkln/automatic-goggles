"""
Dense Retrieval with FAISS

Retrieves similar past screenshots based on their vision token embeddings.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import torch
import faiss

logger = logging.getLogger(__name__)


class DenseRetrieval:
    """
    FAISS-backed dense retrieval for finding similar screenshots.
    
    Retrieves top-k similar past screens based on L2 or cosine distance
    in the embedding space.
    """
    
    def __init__(
        self,
        embedding_dim: int = 4096,
        metric: str = "L2",  # "L2" or "cosine"
        use_gpu: bool = False,
    ):
        """
        Initialize dense retrieval index.
        
        Args:
            embedding_dim: Dimension of embeddings (4096 for Granite-7B)
            metric: Distance metric ("L2" or "cosine")
            use_gpu: Whether to use GPU acceleration (requires faiss-gpu)
        """
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Create FAISS index
        if metric == "L2":
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif metric == "cosine":
            # Cosine uses L2 on normalized vectors
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.embeddings = None
        self.metadata = []
    
    def add(self, embedding: np.ndarray, metadata: dict) -> None:
        """
        Add a single embedding to the index.
        
        Args:
            embedding: (D,) embedding vector
            metadata: Dictionary with screenshot info
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Normalize if using cosine
        if self.metric == "cosine":
            embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-10)
        
        embedding = embedding.astype(np.float32)
        
        self.index.add(embedding)
        self.metadata.append(metadata)
    
    def add_batch(self, embeddings: np.ndarray, metadatas: List[dict]) -> None:
        """
        Add multiple embeddings.
        
        Args:
            embeddings: (N, D) embedding matrix
            metadatas: List of metadata dicts
        """
        embeddings = embeddings.astype(np.float32)
        
        # Normalize if using cosine
        if self.metric == "cosine":
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        
        self.index.add(embeddings)
        self.metadata.extend(metadatas)
        
        logger.debug(f"Added {len(embeddings)} embeddings to index (total: {self.index.ntotal})")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = None,
    ) -> List[Tuple[int, float, dict]]:
        """
        Search for top-k similar embeddings.
        
        Args:
            query_embedding: (D,) query embedding
            k: Number of results to return
            threshold: Optional distance threshold for filtering
        
        Returns:
            List of (index, distance, metadata) tuples
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Normalize if using cosine
        if self.metric == "cosine":
            query_embedding = query_embedding / (
                np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-10
            )
        
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # Invalid index
                continue
            if threshold is not None and dist > threshold:
                continue
            
            results.append((int(idx), float(dist), self.metadata[idx]))
        
        return results
    
    def clear(self) -> None:
        """Clear index and metadata."""
        # Reset index
        if self.metric == "L2":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.metadata = []
        logger.info("Cleared index")
    
    def size(self) -> int:
        """Return number of indexed embeddings."""
        return self.index.ntotal
    
    def save(self, path: str) -> None:
        """Save index to disk."""
        import pickle
        
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
        
        faiss.write_index(cpu_index, path)
        
        # Save metadata separately
        with open(path + ".meta", "wb") as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Saved index to {path}")
    
    def load(self, path: str) -> None:
        """Load index from disk."""
        import pickle
        
        cpu_index = faiss.read_index(path)
        
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index
        
        # Load metadata
        with open(path + ".meta", "rb") as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Loaded index from {path} ({self.size()} vectors)")
