from dotenv import dotenv_values
import time
from pinecone import Pinecone, ServerlessSpec
import uuid
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Load environment variables

config = dotenv_values(".env")

# Configure Pinecone

PINECONE_API_KEY = config["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("chat-history")

# Configure Gemini

GEMINI_API_KEY = config["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# Configure Sentence Transformers

st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_embedding(text):
    embed = st_model.encode(text)
    return embed


def store_memory(user_id, user_message, bot_response):
    user_embed = get_embedding(user_message)
    bot_embed = get_embedding(bot_response)

    user_msg_id = str(uuid.uuid4())
    bot_msg_id = str(uuid.uuid4())

    user_metadata = {
        "username": user_id,
        "message_type": "user",
        "message": user_message,
        "timestamp": int(time.time())
    }

    bot_metadata = {
        "username": "Chikko",
        "message_type": "bot",
        "message": bot_response,
        "timestamp": int(time.time())
        }

    index.upsert([
        (user_msg_id, user_embed, user_metadata),
        (bot_msg_id, bot_embed, bot_metadata)
    ])

    # print(f"Stored {user_id}'s message: {user_message}")
    # print(f"Stored response: {bot_response}")

def retrieve_memory(user_id, user_message, top_k=6):
    query_vector = get_embedding(user_message).tolist()

    relevant_results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )["matches"]

    user_related_results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filters={
            "username": user_id
        }
    )["matches"]

    unique_ids = set()
    combined_results = []

    for match in user_related_results + relevant_results:
        if match["id"] not in unique_ids:
            unique_ids.add(match.id)
            combined_results.append(match)

    return combined_results

def chat_with_bot():
    model = genai.GenerativeModel("gemini-pro")
    conversation = model.start_chat()

    while True:
        user_id = input("Enter your name: ")
        user_input = input("You: ")

        if user_input.lower()in ["exit", "quit"]:
            print("Goodbye!")
            break

        context = retrieve_memory(user_id, user_input)
        formatted_context = "\n".join([f"{i+1}. {result.metadata['username']}: {result.metadata['message']}" for i, result in enumerate(context)])

        prompt = f"""
            Name: Jess
            Personality Traits: funny, nice, friendly, helpyful, insightful, humorous

            context: {formatted_context}

            user: {user_id}
            user_message: {user_input}

        """

        # print(f"PROMPT: {prompt}")

        response = conversation.send_message(prompt)
        print(f"{response.text}")

        store_memory(user_id, user_input, response.text)

    print("CHAT ENDED")


if __name__ == "__main__":
    chat_with_bot()
