import os
import glob
import time
import chromadb
from chromadb.utils import embedding_functions
from src.llm_client import get_client

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "public_indicators_docs"

class RAGPipeline:
    def __init__(self):
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=self.ef
        )
        if self.collection.count() == 0:
            self._index_documents()

    def _index_documents(self, docs_dir="data/docs"):
        docs, ids, metas = [], [], []
        idx = 0
        for filepath in glob.glob(os.path.join(docs_dir, "*.txt")):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            # Chunking por parágrafo
            chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
            for chunk in chunks:
                docs.append(chunk)
                ids.append(f"doc_{idx}")
                metas.append({"source": os.path.basename(filepath)})
                idx += 1
        if docs:
            self.collection.add(documents=docs, ids=ids, metadatas=metas)
            print(f"[OK] {len(docs)} chunks indexados no ChromaDB.")

    def query(self, question: str, model: str = "gpt-4o-mini", provider: str = "openai", top_k: int = 4) -> dict:
        start = time.perf_counter()
        results = self.collection.query(query_texts=[question], n_results=top_k)
        retrieved_chunks = results["documents"][0] if results["documents"] else []
        context = "\n---\n".join(retrieved_chunks)

        llm = get_client(provider)
        prompt = (
            f"Contexto recuperado:\n{context}\n\n"
            f"Pergunta: {question}\n\n"
            "Responda usando APENAS o contexto acima. Se o dado exato não estiver no contexto, "
            "diga explicitamente que não há dados disponíveis. Se responder com número, termine com 'VALOR: <numero>'."
        )
        resp = llm.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = time.perf_counter() - start
        return {
            "answer": resp.choices[0].message.content,
            "latency_s": round(latency, 3),
            "steps": [{"retrieved_chunks": retrieved_chunks}]
        }