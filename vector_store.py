import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []

    def add(self, embeddings, chunks):
        self.index.add(np.array(embeddings).astype("float32"))
        self.text_chunks.extend(chunks)

    def search(self, query_embedding, top_k=3):
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), top_k
        )
        results = [(self.text_chunks[i], distances[0][pos]) for pos, i in enumerate(indices[0])]
        return results
