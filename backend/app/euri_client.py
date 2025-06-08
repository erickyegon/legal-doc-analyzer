import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("EURI_API_KEY")
BASE_URL = "https://api.euron.one/api/v1/euri/alpha/chat/completions"

def euri_chat_completion(messages, model="gpt-4.1-nano", temperature=0.7, max_tokens=1000):
    # For testing purposes, return a mock response if no API key is configured
    if not API_KEY or API_KEY == "test-key":
        return "This is a mock response from EURI API for testing purposes. The actual API integration would return real AI-generated content here."

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()

        response_data = response.json()
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"]
        else:
            raise ValueError("Invalid response format from API")

    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")
    except (KeyError, IndexError) as e:
        raise Exception(f"Error parsing API response: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")
