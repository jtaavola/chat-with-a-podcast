import csv
import os
import tiktoken
import nltk
import pandas

from openai import OpenAI

GPT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Download the required NLTK data
nltk.download("punkt")


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def chunk_text(text: str, chunk_size: int = 300) -> list[str]:
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


def create_embedding(chunk: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """Return an embedding created by OpenAI's embedding API"""
    res = openai.embeddings.create(input=chunk, model=model)
    return res.data[0].embedding


def create_all_embeddings(
    chunks: list[str], model: str = EMBEDDING_MODEL
) -> pandas.DataFrame:
    """
    Return a dataframe with columns `text` and `embedding` that represents all of the
    embeddings.
    """
    texts = []
    embeddings = []

    for chunk in chunks:
        embedding = create_embedding(chunk, model)

        texts.append(chunk)
        embeddings.append(embedding)

    return pandas.DataFrame(data={"text": texts, "embedding": embeddings})


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
