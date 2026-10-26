from time import perf_counter
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
        total_start = perf_counter()

        print("[1/4] Resolving source directory...", flush=True)
        source_dir = Path(directory or self.config.data_dir)

        print(f"[2/4] Loading documents from {source_dir}...", flush=True)
        start = perf_counter()
        documents = load_directory(source_dir)
        print(
            f"[DONE] Loaded {len(documents)} documents "
            f"in {perf_counter() - start:.2f}s",
            flush=True,
        )

        print("[3/4] Removing stale indexes...", flush=True)
        for strategy in ("sentence_window", "auto_merging"):
            stale_index = index_directory(strategy)
            if stale_index.is_dir():
                shutil.rmtree(stale_index)
                print(f"[DONE] Removed {strategy} index", flush=True)

        self.documents = documents
        self.indexes = {}

        print("[4/4] Building and persisting direct index...", flush=True)
        start = perf_counter()
        direct_index = build_and_persist_index(documents, "direct")
        print(
            f"[DONE] Direct index built in {perf_counter() - start:.2f}s",
            flush=True,
        )

        self.indexes = {"direct": direct_index}

        print(f"[DONE] Total ingestion: {perf_counter() - total_start:.2f}s", flush=True)

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

        print("Code run passed this 1")

        return self.ingest_directory(destination)

    def get_index(self, strategy: Strategy) -> VectorStoreIndex:
        if strategy not in self.indexes:
            if self.documents:
                self.indexes[strategy] = build_and_persist_index(
                    self.documents,
                    strategy,
                )
            else:
                self.indexes[strategy] = load_persisted_index(strategy)

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
        query_started = perf_counter()
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

                print(
                    f"[QUERY] Retrieved {len(sources)} sources and prepared "
                    f"{model} in {perf_counter() - query_started:.2f}s",
                    flush=True,
                )

                if attempt:
                    yield f"Primary model unavailable; using {model}.\n\n", sources

                answer_text = ""
                stream_started = perf_counter()
                first_token = True
                for token in response.response_gen:
                    if first_token:
                        print(
                            f"[QUERY] First token from {model} after "
                            f"{perf_counter() - stream_started:.2f}s",
                            flush=True,
                        )
                        first_token = False
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
