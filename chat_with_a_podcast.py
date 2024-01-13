import os
import tiktoken
import nltk

from openai import OpenAI

GPT_MODEL = "gpt-3.5-turbo"

# Download the required NLTK data
nltk.download("punkt")


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def chunk_text(text: str, chunk_size: int = 300):
    """Return the text in token chunks of roughly size `chunk_size`. Keep sentences together."""
    chunks = []
    sentences: list[str] = nltk.sent_tokenize(text)

    while sentences:
        current_chunk = ""
        current_size = 0
        while current_size < chunk_size and sentences:
            sentence = sentences.pop(0)
            current_chunk += f" {sentence}"
            current_size += num_tokens(sentence)
        chunks.append(current_chunk)
    return chunks


openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
response = openai.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "Tell me a joke.",
        },
    ],
    model=GPT_MODEL,
    temperature=0,
)

print(response.choices[0].message.content)
