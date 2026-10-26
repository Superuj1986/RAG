
import argparse

from reading.evaluation import evaluate_dataset
from reading.pipeline import RAGEngine


def main():
    parser = argparse.ArgumentParser(
        description="Advanced RAG Learning Lab",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument(
        "--directory",
        default=None,
        help="Directory containing PDFs",
    )

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument(
        "--strategy",
        choices=["direct", "sentence_window", "auto_merging"],
        default="direct",
    )

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--dataset", default=None)

    args = parser.parse_args()
    engine = RAGEngine()

    if args.command == "ingest":
        result = engine.ingest_directory(args.directory)
        print(result)

    elif args.command == "ask":
        result = engine.ask(args.question, args.strategy)
        print("\nANSWER\n")
        print(result["answer"])

        print("\nSOURCES\n")
        for source in result["sources"]:
            print(
                f"{source['file_name']} | "
                f"page={source['page_label']} | "
                f"score={source['score']}"
            )
            print(source["text"][:500])
            print()

    elif args.command == "evaluate":
        results = evaluate_dataset(
            engine,
            dataset_path=args.dataset,
        )

        print(f"Evaluated {len(results)} question-strategy pairs.")
        print("Results saved to eval/results.csv")


if __name__ == "__main__":
    main()
