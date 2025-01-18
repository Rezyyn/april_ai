import os
import requests

import pygame# For playing audio
import tempfile  # To handle temporary audio files

# Define the FastAPI endpoint URLs
CHAT_API_URL = "http://127.0.0.1:8080/chat"
TTS_API_URL = "http://127.0.0.1:8090/tts"


def play_audio(audio_path: str):
    """
    Play audio using pygame.
    """
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Wait until playback finishes
    except Exception as e:
        print(f"Error playing audio: {e}")
    finally:
        pygame.mixer.quit()


def chat_with_april():
    """
    Interact with April via the FastAPI backend, with TTS playback for responses.
    """
    print("Chat with April! Type 'exit' to end the session.\n")
    user_id = "interactive_user"

    while True:
        user_input = input(">> You: ")
        if user_input.lower() == "exit":
            print("April: Goodbye!")
            break

        # Prepare payload for POST request
        payload = {
            "user_message": user_input,
            "user_id": user_id,
        }

        try:
            # Send request to FastAPI backend for chat
            response = requests.post(CHAT_API_URL, json=payload)
            response.raise_for_status()

            # Parse chat response
            data = response.json()
            april_response = data.get("response", "April didn't understand that.")
            print(f"April: {april_response}")

            # Request TTS audio for the response
            tts_payload = {"text": april_response, "voice": "default", "language": "en"}
            tts_response = requests.post(TTS_API_URL, json=tts_payload)
            tts_response.raise_for_status()

            tts_data = tts_response.json()
            audio_path = tts_data.get("audio_path")

            # Play the TTS audio if available
            if audio_path and os.path.exists(audio_path):
                play_audio(audio_path)
            else:
                print("TTS audio not available.")
        except requests.exceptions.RequestException as e:
            print(f"Error: Unable to communicate with April. {e}")

if __name__ == "__main__":
    chat_with_april()
