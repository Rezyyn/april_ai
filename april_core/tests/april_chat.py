import requests

# Define the FastAPI endpoint URL
API_URL = "http://127.0.0.1:8080/chat"

def chat_with_april():
    """
    Interact with April via the FastAPI backend.
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
            # Send request to FastAPI backend
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            # Parse response
            data = response.json()
            april_response = data.get("response", "April didn't understand that.")
            print(f"April: {april_response}")

        except requests.exceptions.RequestException as e:
            print(f"Error: Unable to communicate with April. {e}")

if __name__ == "__main__":
    chat_with_april()
