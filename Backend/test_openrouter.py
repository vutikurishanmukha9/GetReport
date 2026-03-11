import asyncio
import os
from openai import AsyncOpenAI

async def test_openrouter():
    api_key = os.getenv("OPENROUTER_API_KEY")
    # I'll just use the env var from GetReport if available
    if not api_key:
        from app.core.config import settings
        api_key = settings.OPENROUTER_API_KEY

    print("OpenRouter Key Present:", bool(api_key))
    if not api_key:
        return

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "http://localhost:3000", "X-Title": "GetReport"}
    )
    
    print("\n[1] Testing Embedding route on OpenRouter (if any)...")
    try:
        # OpenRouter doesn't have free embedding models. We'll see what it does.
        # Deepseek-v3 or Llama don't do embeddings through OpenRouter easily, or it costs money.
        res = await asyncio.wait_for(client.embeddings.create(
            input=["What is the trend in sales?"],
            model="text-embedding-3-small"
        ), timeout=30.0)
        print("Embedding Success:", len(res.data[0].embedding))
    except Exception as e:
        print("Embedding Error:", type(e).__name__, str(e))

    print("\n[2] Testing Chat route with Mistral-7b free...")
    try:
        res = await asyncio.wait_for(client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[{"role": "user", "content": "What is the trend in sales?"}],
            max_tokens=100
        ), timeout=30.0)
        print("Chat Success:", res.choices[0].message.content)
    except Exception as e:
        print("Chat Error:", type(e).__name__, str(e))

if __name__ == "__main__":
    asyncio.run(test_openrouter())
