
import csv
import json
from pathlib import Path

from reading.config import get_settings
from reading.indexing import Strategy
from reading.pipeline import RAGEngine


def _score_and_reason(result):
    if isinstance(result, tuple):
        score, reason = result
        return float(score), str(reason)

    return float(result), ""


def evaluate_case(
    engine: RAGEngine,
    question: str,
    strategy: Strategy,
    provider,
) -> dict:
    result = engine.ask(question, strategy)

    context = "\n\n".join(
        source["text"] for source in result["sources"]
    )

    if not context.strip():
        raise ValueError("No retrieved context is available for evaluation.")

    answer_score, answer_reason = _score_and_reason(
        provider.relevance_with_cot_reasons(
            question,
            result["answer"],
        )
    )

    context_score, context_reason = _score_and_reason(
        provider.relevance_with_cot_reasons(
            question,
            context,
        )
    )

    groundedness_score, groundedness_reason = _score_and_reason(
        provider.groundedness_measure_with_cot_reasons(
            source=context,
            statement=result["answer"],
        )
    )

    return {
        "question": question,
        "strategy": strategy,
        "answer": result["answer"],
        "answer_relevance": answer_score,
        "context_relevance": context_score,
        "groundedness": groundedness_score,
        "answer_relevance_reason": answer_reason,
        "context_relevance_reason": context_reason,
        "groundedness_reason": groundedness_reason,
    }


def evaluate_dataset(
    engine: RAGEngine,
    dataset_path: str | Path | None = None,
    strategies: tuple[Strategy, ...] = (
        "direct",
        "sentence_window",
        "auto_merging",
    ),
) -> list[dict]:
    config = get_settings()
    path = Path(dataset_path or config.eval_dataset)

    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    if not config.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for TruLens LLM-judged evaluation."
        )

    from trulens.providers.openai import OpenAI

    provider = OpenAI(
        model_engine=config.openai_model,
        api_key=config.get_openai_api_key(),
    )

    with path.open(encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")

    results = []

    for item in dataset:
        question = item.get("question", "").strip()

        if not question:
            continue

        for strategy in strategies:
            results.append(
                evaluate_case(
                    engine=engine,
                    question=question,
                    strategy=strategy,
                    provider=provider,
                )
            )

    if not results:
        raise ValueError("No valid evaluation questions were found.")

    output_path = path.parent / "results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    return results
