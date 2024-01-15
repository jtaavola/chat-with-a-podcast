import csv
import os
import ast
import tiktoken
import nltk

import pandas
from pandas.core.api import DataFrame

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


def create_all_embeddings(chunks: list[str], model: str = EMBEDDING_MODEL) -> DataFrame:
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

    return DataFrame(data={"text": texts, "embedding": embeddings})


def read_embeddings_from_csv(embeddings_file: str) -> DataFrame:
    """Return a dataframe of embeddings from a csv file."""
    df = pandas.read_csv(embeddings_file)

    # convert embeddings from CSV str type back to list type
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    return df


def write_embeddings_to_csv(df: DataFrame, file_path: str) -> None:
    """Write a dataframe of embeddings to a csv file"""
    df.to_csv(file_path, index=False, quoting=csv.QUOTE_ALL)
