import openai
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

api_key = "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"
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
