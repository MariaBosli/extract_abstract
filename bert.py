from summarizer import Summarizer

def summarize_text(body: str) -> str:
    model = Summarizer()
    summary = model(body)
    return summary
