from __future__ import annotations

import logging
from typing import Any

from universal_ai.models import ChatRequest, Message, MessageRole

logger = logging.getLogger("universal_ai.rag")


class Document:
    def __init__(
        self, content: str, source: str = "", metadata: dict[str, Any] | None = None
    ) -> None:
        self.content = content
        self.source = source
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Document(source={self.source!r}, chars={len(self.content)})"


class Chunk:
    def __init__(
        self,
        content: str,
        document: str = "",
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        self.content = content
        self.document = document
        self.metadata = metadata or {}
        self.embedding = embedding

    def __repr__(self) -> str:
        return f"Chunk(document={self.document!r}, chars={len(self.content)})"


class ChunkingStrategy:
    def chunk(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        raise NotImplementedError


class SentenceChunking(ChunkingStrategy):
    def chunk(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) > chunk_size and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current += " " + sentence
        if current.strip():
            chunks.append(current.strip())
        return chunks


class RecursiveChunking(ChunkingStrategy):
    def chunk(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        import re

        chunks = []
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            if len(para) <= chunk_size:
                chunks.append(para.strip())
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sentence in sentences:
                    if len(current) + len(sentence) > chunk_size and current:
                        chunks.append(current.strip())
                        current = sentence
                    else:
                        current += " " + sentence
                if current.strip():
                    chunks.append(current.strip())
        return chunks


class VectorStore:
    def __init__(self) -> None:
        self._vectors: dict[int, Chunk] = {}
        self._counter: int = 0

    def add(self, chunk: Chunk, vector: list[float]) -> None:
        chunk.embedding = vector
        self._counter += 1
        self._vectors[self._counter] = chunk

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        if not self._vectors:
            return []
        chunks_with_scores: list[tuple[float, Chunk]] = []
        for chunk in self._vectors.values():
            score = self._simple_similarity(query, chunk.content)
            chunks_with_scores.append((score, chunk))
        chunks_with_scores.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in chunks_with_scores[:top_k]]

    def _simple_similarity(self, query: str, text: str) -> float:
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        if not query_words:
            return 0.0
        intersection = query_words & text_words
        return len(intersection) / len(query_words)

    def clear(self) -> None:
        self._vectors.clear()


class SQLiteVectorStore(VectorStore):
    def __init__(self, db_path: str = ":memory:") -> None:
        super().__init__()
        self._db_path = db_path


class MemoryVectorStore(VectorStore):
    pass


class DocumentLoader:
    def load(self, path: str) -> list[Document]:
        raise NotImplementedError


class TextLoader(DocumentLoader):
    def load(self, path: str) -> list[Document]:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        return [Document(content=content, source=path)]


class DirectoryLoader(DocumentLoader):
    def __init__(self, glob_pattern: str = "**/*.txt") -> None:
        self.glob_pattern = glob_pattern

    def load(self, path: str) -> list[Document]:
        import glob

        documents: list[Document] = []
        for file_path in glob.glob(f"{path}/{self.glob_pattern}", recursive=True):
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                documents.append(Document(content=content, source=file_path))
            except Exception:
                continue
        return documents


class RAG:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
        chunking_strategy: ChunkingStrategy | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.chunking_strategy = chunking_strategy or RecursiveChunking()
        self.vector_store = vector_store or MemoryVectorStore()

    def add_text(self, text: str, source: str = "") -> None:
        chunks = self.chunking_strategy.chunk(text, self.chunk_size, self.chunk_overlap)
        for i, chunk_text in enumerate(chunks):
            chunk = Chunk(content=chunk_text, document=source, metadata={"index": i})
            self.vector_store.add(chunk, [])

    def add_document(self, document: Document) -> None:
        self.add_text(document.content, document.source)

    def add_folder(self, path: str) -> None:
        loader = DirectoryLoader()
        documents = loader.load(path)
        for doc in documents:
            self.add_document(doc)

    def add_url(self, url: str) -> None:
        import httpx

        try:
            resp = httpx.get(url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            content = resp.text
            self.add_text(content, source=url)
        except Exception as e:
            logger.warning("Failed to load URL %s: %s", url, e)

    def add_github(self, repo: str) -> None:
        import httpx

        try:
            api_url = f"https://api.github.com/repos/{repo}/contents/"
            resp = httpx.get(api_url, timeout=30.0)
            resp.raise_for_status()
            files = resp.json()
            for file_info in files:
                if file_info.get("type") == "file" and file_info.get("name", "").endswith(
                    (".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".yml")
                ):
                    download_url = file_info.get("download_url")
                    if download_url:
                        file_resp = httpx.get(download_url, timeout=30.0)
                        file_resp.raise_for_status()
                        self.add_text(file_resp.text, source=f"{repo}/{file_info['name']}")
        except Exception as e:
            logger.warning("Failed to load GitHub repo %s: %s", repo, e)

    def add_pdf(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
                content = f.read()
            text = content.decode("utf-8", errors="ignore")
            self.add_text(text, source=path)
        except Exception as e:
            logger.warning("Failed to load PDF %s: %s", path, e)

    async def search(self, query: str) -> list[Chunk]:
        return self.vector_store.search(query, self.top_k)

    async def get_knowledge_context(self, query: str) -> str:
        chunks = await self.search(query)
        return "\n\n".join(chunk.content for chunk in chunks)

    async def augment_request(self, request: ChatRequest) -> ChatRequest:
        if not request.messages:
            return request
        last_message = request.messages[-1].content if request.messages else ""
        context = await self.get_knowledge_context(last_message)
        if context:
            augmented = f"Relevant context: {context}\n\nUser question: {last_message}"
            new_messages = list(request.messages[:-1])
            new_messages.append(Message(role=MessageRole.user, content=augmented))
            return ChatRequest(
                messages=new_messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=request.tools,
                stream=request.stream,
            )
        return request
