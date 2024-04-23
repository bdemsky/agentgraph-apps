import faiss
import nltk
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

class Retriever:

    def __init__(self, documents: list[str]):
        self.documents = documents

    def tokenize(self, text: str):
        return nltk.word_tokenize(text)

    def encode(self, text: str):
        return self.tokenize(text)

    def encode_all(self, text: list[str]):
        return [self.encode(s) for s in text]

    def retrieve(self, query: str, k: int):
        pass

class FAISSRetriever(Retriever):

    def __init__(self, index_path: str, override: bool, device, *args):
        super().__init__(*args)
        # avoid taking up the gpu
        self.transformer = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.index = None 
        if not override:
            self.index = self.load_index(index_path)
            if self.index is not None and self.index.ntotal != len(self.documents):
                print(f"incompatible index dimensions: loaded {self.index.ntotal} but needs {len(self.documents)}")
                self.index = None

        if self.index is None:
            encoded_docs = self.encode_all(self.documents)
            self.index = faiss.IndexFlatL2(encoded_docs.shape[1])
            self.index.add(encoded_docs)
            if index_path is not None:
                self.save_index(index_path)

    def encode(self, text: str):
        return self.transformer.encode(text).reshape(1, -1)
    
    def encode_all(self, text: list[str]):
        return self.transformer.encode(text)

    def retrieve(self, query: str, k: int):
        encoded_q = self.encode(query)
        _, q_ids = self.index.search(encoded_q, k)
        q_ids = q_ids[0]

        if (q_ids == -1).any():
            print(f"Insufficient documents for top-{k} docs")

        return [self.documents[q_id] for q_id in q_ids]

    def load_index(self, filepath: str) -> 'faiss.IndexFlatL2':
        if filepath is not None and os.path.exists(filepath):
            index = faiss.read_index(filepath)
            print(f"Loaded index from '{filepath}' with {index.ntotal} embeddings.")
            return index

        return None

    def save_index(self, filepath: str):
        # create directory if it doesn't exist
        os.makedirs(Path(filepath).parent, exist_ok=True)

        print("Persisting the index at", filepath)
        faiss.write_index(self.index, filepath)

