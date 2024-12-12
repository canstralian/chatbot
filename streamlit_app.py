import streamlit as st
from transformers import pipeline

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses Hugging Face's models to generate responses. "
    "You can choose from a variety of models available on the Hugging Face model hub."
)

# You can also add the option for the user to select a model, for example:
model_name = st.selectbox("Select a model:", [
    "gpt2", "distilgpt2", "EleutherAI/gpt-neo-125M"
])

try:
    # Create a Hugging Face pipeline for text generation
    @st.cache_resource
    def load_model(model_name):
        return pipeline("text-generation", model=model_name)
    
    generator = load_model(model_name)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What is up?"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response using the Hugging Face model.
    response = generator(prompt, max_length=150, num_return_sequences=1)

    # The response is a list of dictionaries, so we extract the text.
    assistant_response = response[0]['generated_text']

    # Display the assistant's response and store it in session state.
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_response
    })
