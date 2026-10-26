
import sys
from pathlib import Path
from time import perf_counter

import gradio as gr
import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from reading import RAGEngine  # noqa: E402
from reading.indexing import configure_models  # noqa: E402

engine = RAGEngine()
configure_models()


@spaces.GPU(duration=120)
def ingest(files):
    if not files:
        return "Upload at least one PDF."

    try:
        result = engine.ingest_uploaded_files(files)
        return (
            "Ingestion completed.\n\n"
            f"Documents/pages loaded: {result['documents']}\n\n"
            "Finish indexing document. Please select Ask Question tab to ask question"
        )
    except Exception as exc:
        return f"Ingestion failed: {exc}"


@spaces.GPU(duration=120)
def answer(question, strategy):
    try:
        started = perf_counter()
        for answer_text, sources in engine.ask_stream(question, strategy):
            source_parts = []
            for i, source in enumerate(sources, start=1):
                score = (
                    f"{source['score']:.4f}"
                    if source["score"] is not None
                    else "N/A"
                )
                source_parts.append(
                    f"### Source {i}\n"
                    f"- File: {source['file_name']}\n"
                    f"- Page: {source['page_label']}\n"
                    f"- Retrieval score: {score}\n\n"
                    f"{source['text']}"
                )
            yield answer_text, "\n\n---\n\n".join(source_parts)
        print(f"[QUERY] Completed in {perf_counter() - started:.2f}s", flush=True)

    except Exception as exc:
        yield f"Error: {exc}", ""


with gr.Blocks(title="Advanced RAG Learning Lab") as demo:
    gr.Markdown(
        """
        # Advanced RAG Learning Lab

        Upload PDF documents, compare retrieval strategies, and inspect
        the evidence retrieved for each answer.
        """
    )

    with gr.Tab("1. Documents"):
        pdf_files = gr.File(
            label="Upload PDF documents",
            file_count="multiple",
            file_types=[".pdf"],
            type="filepath",
        )
        ingest_button = gr.Button("Upload and prepare Direct index")
        ingest_status = gr.Textbox(
            label="Ingestion status",
            lines=5,
            interactive=False,
        )

        ingest_button.click(
            fn=ingest,
            inputs=[pdf_files],
            outputs=[ingest_status],
        )

    with gr.Tab("2. Ask questions"):
        strategy = gr.Radio(
            choices=[
                ("Direct RAG", "direct"),
                ("Sentence Window", "sentence_window"),
                ("Auto Merging", "auto_merging"),
            ],
            value="direct",
            label="Retrieval strategy",
        )

        question = gr.Textbox(
            label="Question",
            placeholder="Ask something answered by your documents...",
            lines=3,
        )

        ask_button = gr.Button("Ask")
        answer_output = gr.Markdown(label="Answer")
        source_output = gr.Markdown(label="Retrieved evidence")

        ask_button.click(
            fn=answer,
            inputs=[question, strategy],
            outputs=[answer_output, source_output],
        )

        question.submit(
            fn=answer,
            inputs=[question, strategy],
            outputs=[answer_output, source_output],
        )


if __name__ == "__main__":
    demo.launch()
