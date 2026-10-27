import shutil
from pathlib import Path

from llama_index.core import Document, VectorStoreIndex

from reading.config import get_settings
from reading.generation import (
    configure_llm,
    generate_answer,
    is_service_unavailable,
)
from reading.indexing import (
    Strategy,
    build_and_persist_index,
    index_directory,
    load_persisted_index,
)
from reading.ingestion import load_directory, load_pdfs
from reading.retrieval import build_query_engine


class RAGEngine:
    def __init__(self):
        self.config = get_settings()
        self.indexes: dict[str, VectorStoreIndex] = {}
        self.documents: list[Document] = []

    def ingest_directory(self, directory: str | Path | None = None):
        source_dir = Path(directory or self.config.data_dir)

        documents = load_directory(source_dir)

        for strategy in ("sentence_window", "auto_merging"):
            stale_index = index_directory(strategy)
            if stale_index.is_dir():
                shutil.rmtree(stale_index)

        self.documents = documents
        self.indexes = {}

        direct_index = build_and_persist_index(documents, "direct")

        self.indexes = {"direct": direct_index}

        return {
            "documents": len(documents),
            "indexes": list(self.indexes),
            "source_directory": str(source_dir),
        }

    def load_uploaded_files(self, file_paths):
        if not file_paths:
            raise ValueError("Upload at least one PDF file.")

        source_paths = [Path(file_path) for file_path in file_paths]
        loaded_documents = load_pdfs(source_paths)
        file_names = [path.name.casefold() for path in source_paths]
        if len(file_names) != len(set(file_names)):
            raise ValueError("Uploaded PDFs must have unique file names.")

        destination = Path(self.config.data_dir)
        destination.mkdir(parents=True, exist_ok=True)

        # A new upload replaces the active corpus; remove stale source PDFs
        # and indexes so later queries cannot retrieve from an earlier upload.
        uploaded_paths = {path.resolve() for path in source_paths}
        for existing_pdf in destination.glob("*.pdf"):
            if existing_pdf.resolve() not in uploaded_paths:
                existing_pdf.unlink()
        for strategy in ("direct", "sentence_window", "auto_merging"):
            stale_index = index_directory(strategy)
            if stale_index.is_dir():
                shutil.rmtree(stale_index)

        for source in source_paths:
            target = destination / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)

        self.documents = loaded_documents
        self.indexes = {}
        return {
            "documents": len(loaded_documents),
            "files": len(source_paths),
            "source_directory": str(destination),
        }

    def get_index(self, strategy: Strategy) -> VectorStoreIndex:
        # A new upload removes persisted indexes. This also invalidates cached
        # indexes held by a separate ZeroGPU callback worker.
        if strategy in self.indexes and not (
            index_directory(strategy) / "docstore.json"
        ).exists():
            self.indexes.pop(strategy)

        if strategy not in self.indexes:
            # Use freshly loaded documents first. In a separate ZeroGPU
            # callback worker, recover PDFs and build this strategy when its
            # persisted index is missing.
            if self.documents:
                self.indexes[strategy] = build_and_persist_index(
                    self.documents,
                    strategy,
                )
            else:
                try:
                    self.indexes[strategy] = load_persisted_index(strategy)
                except FileNotFoundError:
                    self.documents = load_directory(self.config.data_dir)
                    self.indexes[strategy] = build_and_persist_index(
                        self.documents,
                        strategy,
                    )

        return self.indexes[strategy]

    def ask(self, question: str, strategy: Strategy = "direct") -> dict:
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        index = self.get_index(strategy)
        llm = configure_llm()

        query_engine = build_query_engine(
            index=index,
            strategy=strategy,
            llm=llm,
        )

        response = generate_answer(query_engine, question)
        answer_text = "".join(response.response_gen)

        sources = []

        for source_node in response.source_nodes:
            metadata = source_node.node.metadata

            sources.append(
                {
                    "text": source_node.node.get_content()[:1200],
                    "score": source_node.score,
                    "file_name": metadata.get("file_name", "Unknown"),
                    "page_label": metadata.get("page_label", "Unknown"),
                }
            )

        return {
            "question": question,
            "strategy": strategy,
            "answer": answer_text,
            "sources": sources,
        }

    def ask_stream(self, question: str, strategy: Strategy = "direct"):
        """Yield the growing answer and its retrieved sources as tokens arrive."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        config = get_settings()
        models = list(
            dict.fromkeys((config.gemini_model, config.gemini_fallback_model))
        )
        index = self.get_index(strategy)

        for attempt, model in enumerate(models):
            try:
                query_engine = build_query_engine(
                    index=index,
                    strategy=strategy,
                    llm=configure_llm(model),
                )
                response = generate_answer(query_engine, question)

                sources = []
                for source_node in response.source_nodes:
                    metadata = source_node.node.metadata
                    sources.append(
                        {
                            "text": source_node.node.get_content()[:1200],
                            "score": source_node.score,
                            "file_name": metadata.get("file_name", "Unknown"),
                            "page_label": metadata.get("page_label", "Unknown"),
                        }
                    )

                if attempt:
                    yield f"Primary model unavailable; using {model}.\n\n", sources

                answer_text = ""
                for token in response.response_gen:
                    answer_text += token
                    yield answer_text, sources
                return
            except Exception as exc:
                if attempt + 1 >= len(models) or not is_service_unavailable(exc):
                    raise
                print(
                    f"[WARN] Gemini model {model} unavailable; falling back to "
                    f"{models[attempt + 1]}: {exc}",
                    flush=True,
                )
