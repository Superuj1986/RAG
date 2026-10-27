
import sys
from pathlib import Path

import gradio as gr
import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from reading import RAGEngine  # noqa: E402

engine = RAGEngine()


def load_documents(files):
    if not files:
        return "Upload at least one PDF."

    try:
        result = engine.load_uploaded_files(files)
        return (
            "Documents loaded. No index has been built yet.\n\n"
            f"PDF files: {result['files']}\n"
            f"Pages loaded: {result['documents']}\n\n"
            "Choose a retrieval strategy, enter a question, and select Ask."
        )
    except Exception as exc:
        return f"Document loading failed: {exc}"


@spaces.GPU(duration=120)
def answer(question, strategy):
    try:
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

    gr.Markdown("## 1. Upload documents")
    pdf_files = gr.File(
        label="Upload PDF documents",
        file_count="multiple",
        file_types=[".pdf"],
        type="filepath",
    )
    ingest_status = gr.Textbox(
        label="Document loading status",
        lines=3,
        interactive=False,
    )
    pdf_files.upload(
        fn=load_documents,
        inputs=[pdf_files],
        outputs=[ingest_status],
    )

    gr.Markdown("## 2. Choose a strategy and ask")
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

    gr.Markdown("## 3. Answer and retrieved sources")
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
