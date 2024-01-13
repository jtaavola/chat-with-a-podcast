import os
from openai import OpenAI

GPT_MODEL = "gpt-3.5-turbo"

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
