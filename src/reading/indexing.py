from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from reading.config import get_settings
from reading.ingestion import load_directory, load_pdfs

Strategy = Literal["direct", "sentence_window", "auto_merging"]


@lru_cache(maxsize=1)
def configure_models() -> None:
    config = get_settings()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config.embed_model,
        device=device,
        embed_batch_size=64,
        show_progress_bar=False,
    )


def build_index(
    documents,
    strategy: Strategy,
) -> VectorStoreIndex:
    configure_models()

    if strategy == "direct":
        parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50,
        )

    elif strategy == "sentence_window":
        parser = SentenceWindowNodeParser.from_defaults(
            window_size=get_settings().sentence_window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

    elif strategy == "auto_merging":
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128],
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    nodes = parser.get_nodes_from_documents(documents)

    if not nodes:
        raise ValueError("The parser produced no nodes.")

    index = VectorStoreIndex(nodes)
    return index


def index_directory(strategy: Strategy) -> Path:
    return Path(get_settings().index_dir) / strategy


def build_and_persist_index(
    documents,
    strategy: Strategy,
    persist_dir: str | Path | None = None,
) -> VectorStoreIndex:
    index = build_index(documents, strategy)

    destination = Path(persist_dir) if persist_dir else index_directory(strategy)
    destination.mkdir(parents=True, exist_ok=True)

    index.storage_context.persist(persist_dir=str(destination))

    return index


def load_persisted_index(
    strategy: Strategy,
    persist_dir: str | Path | None = None,
) -> VectorStoreIndex:
    configure_models()

    source = Path(persist_dir) if persist_dir else index_directory(strategy)

    if not (source / "docstore.json").exists():
        raise FileNotFoundError(
            f"No persisted index found at {source}. "
            "Ingest PDFs and build the index first."
        )

    storage_context = StorageContext.from_defaults(
        persist_dir=str(source),
    )

    return load_index_from_storage(storage_context)


def build_all_indexes(documents) -> dict[str, VectorStoreIndex]:
    return {
        strategy: build_and_persist_index(documents, strategy)
        for strategy in ("direct", "sentence_window", "auto_merging")
    }


def ingest_directory(directory: str | Path | None = None):
    config = get_settings()
    documents = load_directory(directory or config.data_dir)
    indexes = build_all_indexes(documents)

    return documents, indexes


def ingest_files(paths):
    documents = load_pdfs(paths)
    indexes = build_all_indexes(documents)

    return documents, indexes
