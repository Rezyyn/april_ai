import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Logging configuration
logging.basicConfig(level=logging.DEBUG)  # Default level is DEBUG for development
logger = logging.getLogger(__name__)

# Function to configure logging level dynamically
def set_log_level(level: str):
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logger.setLevel(levels.get(level.lower(), logging.DEBUG))
    logger.info(f"Log level set to: {level.upper()}")

# Set initial log level
set_log_level("info")

# Load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Optimize GPU inference speed
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
    version="1.0.0"
)

class ChatRequest(BaseModel):
    user_message: str
    user_id: Optional[str] = None
    live2d_flag: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    live2d_command: Optional[str] = None

SYSTEM_PROMPT = (
    "System: You are April, a whimsical, self-absorbed adult with a playful, childlike personality. "
    "You are fascinated by absurd and nonsensical phrases, often erupting into chaotic, repetitive outbursts. "
    "However, you balance your eccentricity with an ability to stay on topic for brief periods. "
    "You occasionally disrupt conversations, particularly when triggered by certain words or ideas, but you always return to the user's focus."
)

conversation_history = {}

# Generate response
def generate_response(user_input: str, user_id: str) -> str:
    if not user_input.strip():  # Handle empty user input
        logger.warning("Empty user input received.")
        return "April is waiting for you to say something!"

    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # Limit the context to the last 3 exchanges for brevity
    context = SYSTEM_PROMPT
    for message in conversation_history[user_id][-3:]:
        context += f"{message}\n"
    context += f"User: {user_input.strip()}\nApril:"

    logger.debug(f"Context for model:\n{context}")

    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=150,  # Increased from 50 to 150 for longer responses
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,  # Ensure generation stops at EOS
                do_sample=True,
                temperature=0.6,  # Lower temperature for coherent responses
                top_p=0.9,        # Higher top-p for diverse yet focused sampling
                repetition_penalty=1.2  # Discourage repetitive phrases
            )

        # Decode the generated response
        response = tokenizer.decode(
            output_ids[:, input_ids.shape[-1]:][0],
            skip_special_tokens=True
        ).strip()

        # Ensure the response ends with proper punctuation
        if response and response[-1] not in ".!?":
            response += "."

        logger.info("Response generated successfully.")
    except Exception as e:
        logger.error(f"Error during response generation: {e}")
        return "April is having trouble thinking right now!"

    # Add user input and response to the history
    conversation_history[user_id].append(f"User: {user_input.strip()}")
    conversation_history[user_id].append(f"April: {response}")

    # Limit the history length to avoid overhead
    if len(conversation_history[user_id]) > 6:
        conversation_history[user_id] = conversation_history[user_id][-6:]

    return response

    if not user_input.strip():  # Handle empty user input
        logger.warning("Empty user input received.")
        return "April is waiting for you to say something!"

    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # Limit the context to the last 3 exchanges for brevity
    context = SYSTEM_PROMPT
    for message in conversation_history[user_id][-3:]:
        context += f"{message}\n"
    context += f"User: {user_input.strip()}\nApril:"

    logger.debug(f"Context for model:\n{context}")

    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=50,  # Limit response length to 50 tokens
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,  # Lower temperature for concise responses
                top_p=0.8,        # Narrow top-p for faster and more focused sampling
                repetition_penalty=1.2  # Discourage repetitive phrases
            )

        response = tokenizer.decode(
            output_ids[:, input_ids.shape[-1]:][0],
            skip_special_tokens=True
        ).strip()

        logger.info("Response generated successfully.")
    except Exception as e:
        logger.error(f"Error during response generation: {e}")
        return "April is having trouble thinking right now!"

    # Add user input and response to the history
    conversation_history[user_id].append(f"User: {user_input.strip()}")
    conversation_history[user_id].append(f"April: {response}")

    # Limit the history length to avoid overhead
    if len(conversation_history[user_id]) > 6:
        conversation_history[user_id] = conversation_history[user_id][-6:]

    return response

def generate_live2d_command(response: str) -> str:
    if "happy" in response.lower():
        return "trigger_happy_expression"
    elif "sad" in response.lower():
        return "trigger_sad_expression"
    elif "angry" in response.lower():
        return "trigger_angry_expression"
    else:
        return "trigger_neutral_expression"

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not req.user_message.strip():
        logger.error("Empty user message in request.")
        raise HTTPException(
            status_code=400, detail="User message cannot be empty."
        )

    user_id = req.user_id or "default_user"

    try:
        april_response = generate_response(req.user_message.strip(), user_id)
    except Exception as e:
        logger.critical(f"Critical error generating response: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

    live2d_command = None
    if req.live2d_flag:
        live2d_command = generate_live2d_command(april_response)

    return ChatResponse(response=april_response, live2d_command=live2d_command)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
