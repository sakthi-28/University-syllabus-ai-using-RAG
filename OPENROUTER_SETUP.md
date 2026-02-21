# OpenRouter Setup Guide

## Using GLM-4.5-Air (Free) via OpenRouter

This application now supports OpenRouter, which provides free access to GLM-4.5-Air and many other models.

### Step 1: Get Your OpenRouter API Key

1. Go to [OpenRouter.ai](https://openrouter.ai/)
2. Sign up for a free account
3. Navigate to [API Keys](https://openrouter.ai/keys)
4. Create a new API key
5. Copy your API key

### Step 2: Configure Your .env File

Edit your `.env` file and set:

```env
LLM_PROVIDER=openrouter
LLM_MODEL=z-ai/glm-4.5-air:free
OPENROUTER_API_KEY=your_actual_api_key_here
```

### Step 3: Run the Application

```bash
streamlit run app.py
```

## Available Models on OpenRouter

You can use any model available on OpenRouter by changing the `LLM_MODEL` in your `.env` file:

- **Free Models:**
  - `z-ai/glm-4.5-air:free` - GLM-4.5-Air (free tier)
  - `google/gemini-flash-1.5:free` - Google Gemini Flash
  - `meta-llama/llama-3.2-3b-instruct:free` - Llama 3.2 3B

- **Paid Models (require credits):**
  - `openai/gpt-4` - GPT-4
  - `anthropic/claude-3-opus` - Claude 3 Opus
  - `z-ai/glm-5` - GLM-5 (newer model)

## Benefits of OpenRouter

- ✅ Free tier available for many models
- ✅ Access to multiple AI providers in one place
- ✅ Easy model switching
- ✅ Competitive pricing for paid models
- ✅ OpenAI-compatible API

## Notes

- The free GLM-4.5-Air model has rate limits
- For production use, consider upgrading to paid models
- OpenRouter uses OpenAI-compatible API, so it works seamlessly with LangChain
