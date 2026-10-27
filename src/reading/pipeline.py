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
from reading.ingestion import load_directory
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

    def ingest_uploaded_files(self, file_paths):
        destination = Path(self.config.data_dir)
        destination.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for file_path in file_paths:
            source = Path(file_path)

            if not source.is_file() or source.suffix.lower() != ".pdf":
                raise ValueError(f"Invalid PDF file: {source}")

            target = destination / source.name

            if source.resolve() != target.resolve():
                shutil.copy2(source, target)

            saved_paths.append(target)

        return self.ingest_directory(destination)

    def get_index(self, strategy: Strategy) -> VectorStoreIndex:
        if strategy not in self.indexes:
            # Gradio/ZeroGPU callbacks may run in a worker that does not share
            # this RAGEngine instance's in-memory state. Prefer a saved index,
            # then rebuild the requested strategy from the PDFs still on disk.
            try:
                self.indexes[strategy] = load_persisted_index(strategy)
            except FileNotFoundError:
                if not self.documents:
                    self.documents = load_directory(self.config.data_dir)
                if not self.documents:
                    raise FileNotFoundError(
                        "No PDFs or saved index are available. Upload PDFs first."
                    )
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
