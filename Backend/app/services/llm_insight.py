from openai import AsyncOpenAI
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Initialize OpenAI Client (Async)
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def generate_insights(summary_stats: dict) -> str:
    """
    Generates natural language insights based on the statistical summary.
    """
    # If no API key is set, return a mock insight to prevent crashing
    if not settings.OPENAI_API_KEY:
        return "AI Insights are unavailable. Please configure OPENAI_API_KEY in .env file."

    try:
        prompt = f"""
        You are a Data Analyst. Analyze the following statistical summary of a dataset and provide 
        3-5 key insights or trends. Keep it professional and concise.
        
        Data Summary:
        {summary_stats}
        """
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini", # Use a cost-effective model
            messages=[
                {"role": "system", "content": "You are a helpful data analyst helper."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error calling OpenAI: {str(e)}")
        return "Could not generate AI insights at this time."
