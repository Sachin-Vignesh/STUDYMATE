import datetime

def save_chat_history(history, filename="chat_history.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for entry in history:
            f.write(f"Q: {entry['question']}\nA: {entry['answer']}\n\n")

def format_sources(sources):
    return "\n\nSources:\n" + "\n".join([src for src, _ in sources])
