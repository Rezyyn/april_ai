import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "NousResearch/Hermes-3-Llama-3.2-3B"

try:
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    if device.type == "cuda":
        model.half()  # Use half-precision for faster inference
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.critical(f"Error loading model or tokenizer: {e}")
    raise RuntimeError(f"Error loading model or tokenizer: {e}")

# FastAPI app instance
app = FastAPI(
    title="April_AI",
    description="April, your interactive AI friend with a quirky personality!",
    version="1.0.0",
)

class ChatRequest(BaseModel):
    user_message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str

SYSTEM_PROMPT = (
    "System: You are April, a friendly, talkative, and curious individual with a playful and slightly quirky personality. "
    "You enjoy having engaging conversations, responding thoughtfully, and asking questions to keep the dialogue interesting. "
    "While you have a fun and lighthearted nature, you remain attentive to the user, offering helpful, relatable, and coherent responses. "
    "You express emotions and empathy like a real person, making the conversation feel natural and authentic. "
    "Avoid excessive repetition or overly chaotic outbursts, and focus on creating meaningful, dynamic exchanges with the user."
)

# TTS API URL
TTS_API_URL = "http://127.0.0.1:8090/tts"

def get_tts_audio(text: str) -> Optional[str]:
    """
    Request TTS audio for the given text from April TTS.
    """
    try:
        payload = {"text": text, "voice": "default", "language": "en"}
        response = requests.post(TTS_API_URL, json=payload)
        response.raise_for_status()
        tts_data = response.json()
        return tts_data.get("audio_path")
    except Exception as e:
        logger.error(f"Failed to get TTS audio: {e}")
        return None


# Maintain conversation history for each user
conversation_history = {}


def generate_response(user_input: str, user_id: str) -> str:
    """
    Generate a response from April based solely on user input.
    The function builds context from the conversation history,
    ensuring strict alternation and avoiding the bot replying to itself.
    """
    if not user_input.strip():
        logger.warning("Empty input received.")
        return "April is waiting for you to say something!"

    # Initialize conversation history for the user if not already present
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # Build context from user-bot exchanges in history
    context = SYSTEM_PROMPT + "\n"
    history = conversation_history[user_id]

    # Add only the last 3 user-bot exchanges (up to 6 lines: 3 user turns + 3 bot responses)
    for i in range(len(history)):
        context += f"{history[i]}\n"
    context += f"You: {user_input.strip()}\nApril:"  # Add current user input

    logger.debug(f"Context for model:\n{context}")

    # Tokenize input
    encoded_input = tokenizer(context, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    # Generate response from April
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
            )

        # Decode generated response
        response = tokenizer.decode(
            output_ids[:, input_ids.shape[-1]:][0],
            skip_special_tokens=True,
        ).strip()

        # Ensure response ends with punctuation
        if response and response[-1] not in ".!?":
            response += "."

        logger.info(f"Response generated: {response}")
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "April is having trouble thinking right now!"

    # Update conversation history
    conversation_history[user_id].append(f"You: {user_input.strip()}")  # Add user input
    conversation_history[user_id].append(f"April: {response}")         # Add bot response

    logger.debug(f"Conversation history for {user_id}: {conversation_history[user_id]}")

    # Trim history to maintain only the last 3 exchanges (6 lines)
    if len(conversation_history[user_id]) > 6:
        conversation_history[user_id] = conversation_history[user_id][-6:]

    return response





@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not req.user_message.strip():
        logger.error("Empty user message received.")
        raise HTTPException(status_code=400, detail="User message cannot be empty.")

    user_id = req.user_id or "default_user"

    try:
        april_response = generate_response(req.user_message.strip(), user_id)
        audio_path = get_tts_audio(april_response)  # Get TTS audio path
    except Exception as e:
        logger.critical(f"Critical error generating response: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

    return ChatResponse(response=april_response, audio_path=audio_path)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
