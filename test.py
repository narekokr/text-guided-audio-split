import os

import openai
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are a helpful assistant that extracts music stems."
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=(
                "From the following user request, extract only the valid stems (vocals, drums, bass, other). "
                "Return them as a comma-separated list and nothing else.\n\n"
                "User request: separate vocals and drums"
            )
        )
    ],
    max_tokens=10,
    temperature=0
)

print(response.choices[0].message.content)