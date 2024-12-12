
import streamlit as st
from transformers import pipeline
from datasets import load_dataset
import time

# Set page configuration
st.set_page_config(
    page_title="💬 AI Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for model and dataset selection
with st.sidebar:
    st.title("🛠️ Configuration")
    model_name = st.selectbox(
        "Select Model:",
        ["distilgpt2", "gpt2", "EleutherAI/gpt-neo-125M"],
        help="Choose the model for text generation"
    )
    
    st.subheader("Dataset Configuration")
    dataset_source = st.text_input(
        "Enter Dataset (e.g. 'microsoft/DialogStudio' or URL):",
        help="Enter a Hugging Face dataset name or URL"
    )
    
    if dataset_source:
        try:
            with st.spinner("Loading dataset..."):
                # Use streaming for large datasets
                dataset = load_dataset(
                    dataset_source,
                    streaming=True,
                    split="train"
                )
                # Take a small sample
                sample_size = 1000
                dataset = dataset.take(sample_size)
                st.success(f"Dataset loaded successfully! (Sample size: {sample_size})")
                
                # Show a preview of the first few examples
                preview = list(dataset.take(3))
                st.write("Preview of dataset:", preview)
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")

# Main chat interface
st.title("💬 AI Chatbot")
st.markdown("""
<style>
    .stChat {
        border-radius: 10px;
        padding: 10px;
    }
    .stTextInput {
        border-radius: 5px;
    }
    .stButton button {
        border-radius: 5px;
        background-color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

st.write(
    "This chatbot uses Hugging Face's models for text generation. "
    "You can customize the experience by selecting different models and datasets."
)

try:
    @st.cache_resource
    def load_model(model_name):
        try:
            # Add specific configurations for text generation
            return pipeline(
                "text-generation",
                model=model_name,
                trust_remote_code=True,
                device=-1,  # Force CPU
                model_kwargs={"low_cpu_mem_usage": True}
            )
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None
    
    with st.spinner(f"Loading {model_name}..."):
        generator = load_model(model_name)
        if generator is not None:
            st.success(f"Model {model_name} loaded successfully!")
        else:
            st.error("Failed to load model. Please try selecting a different model.")
            st.stop()
except Exception as e:
    st.error(f"Error initializing model pipeline: {str(e)}")
    st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat display
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        response = generator(prompt, max_length=150, num_return_sequences=1)
        assistant_response = response[0]['generated_text']

    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_response
    })

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()
