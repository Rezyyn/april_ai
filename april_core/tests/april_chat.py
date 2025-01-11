import requests

# Define the FastAPI endpoint URL
API_URL = "http://127.0.0.1:8080/chat"


def chat_with_api():
    """
    Interactively chat with the FastAPI backend via the `/chat` endpoint.
    """
    print("April is ready! Type 'exit' to end the conversation.\n")

    # User ID for conversation tracking
    user_id = "interactive_user"

    while True:
        # Get user input
        user_input = input(">> You: ")
        if user_input.lower() == "exit":
            print("April: Goodbye!")
            break

        # Prepare the payload for the API request
        payload = {
            "user_message": user_input,
            "user_id": user_id,
            "live2d_flag": False,  # Change to True if Live2D commands are
                                   # needed
        }

        try:
            # Send POST request to the FastAPI /chat endpoint
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Raise an error for HTTP issues
            # Parse the API response
            data = response.json()
            ai_response = data.get(
                "response", "Sorry, I couldn't understand that."
            )

            # Display the bot's response
            print(f"April: {ai_response}")

            # Optionally, display Live2D command
            if data.get("live2d_command"):
                print(f"(Live2D Command: {data['live2d_command']})")

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with the chatbot API: {e}")


if __name__ == "__main__":
    chat_with_api()
