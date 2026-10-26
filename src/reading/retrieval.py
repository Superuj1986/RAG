
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

from reading.config import get_settings
from reading.indexing import Strategy


def build_retriever(index: VectorStoreIndex, strategy: Strategy):
    config = get_settings()

    if strategy == "direct":
        return index.as_retriever(
            similarity_top_k=config.direct_top_k,
        )

    if strategy == "sentence_window":
        return index.as_retriever(
            similarity_top_k=config.sentence_window_top_k,
        )

    if strategy == "auto_merging":
        base_retriever = index.as_retriever(
            similarity_top_k=config.auto_merging_top_k,
        )

        return AutoMergingRetriever(
            base_retriever,
            index.storage_context,
            verbose=False,
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def build_postprocessors(strategy: Strategy):
    config = get_settings()
    processors = []

    if strategy == "sentence_window":
        processors.append(
            MetadataReplacementPostProcessor(
                target_metadata_key="window",
            )
        )

    if config.enable_reranker and strategy != "direct":
        processors.append(
            SentenceTransformerRerank(
                model=config.rerank_model,
                top_n=config.rerank_top_n,
            )
        )

    return processors


def build_query_engine(
    index: VectorStoreIndex,
    strategy: Strategy,
    llm=None,
) -> RetrieverQueryEngine:
    retriever = build_retriever(index, strategy)

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=build_postprocessors(strategy),
        llm=llm,
        response_mode="compact",
        streaming=True,
    )
