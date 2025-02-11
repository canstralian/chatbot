import streamlit as st
from transformers import pipeline
from datasets import load_dataset

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None

def init_page_config():
    """Initialize Streamlit page configuration."""
    st.set_page_config(
        page_title="💬 AI Chatbot",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def setup_sidebar():
    """Configure and return sidebar elements."""
    with st.sidebar:
        st.title("🛠️ Configuration")
        model_name = st.selectbox(
            "Select Model:",
            ("distilgpt2", "gpt2", "EleutherAI/gpt-neo-125M"),
            help="Choose the model for text generation"
        )
        
        load_dataset_config()
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state['messages'] = []
            st.experimental_rerun()
            
        return model_name

def load_dataset_config():
    """Handle dataset loading and configuration."""
    st.subheader("Dataset Configuration")
    dataset_source = st.text_input(
        "Enter Dataset (e.g., 'microsoft/DialogStudio' or URL):",
        help="Enter a Hugging Face dataset name or URL"
    )
    
    if dataset_source:
        try:
            with st.spinner("Loading dataset..."):
                dataset = load_dataset(
                    dataset_source,
                    streaming=True,
                    split="train"
                ).take(1000)
                st.session_state['dataset'] = list(dataset)
                st.success("Dataset loaded successfully! (Sample size: 1000)")
                st.write("Preview of dataset:", st.session_state['dataset'][:3])
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

@st.cache_resource
def load_model(model_name: str):
    """
    Load and cache the language model.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        transformers.pipelines.Pipeline: The loaded text-generation pipeline.
    """
    try:
        return pipeline(
            "text-generation",
            model=model_name,
            trust_remote_code=True,
            device=-1,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def setup_chat_interface():
    """Set up the main chat interface."""
    st.title("💬 AI Chatbot")
    st.markdown("""
        <style>
            .stChat { border-radius: 10px; padding: 10px; }
            .stTextInput { border-radius: 5px; }
            .stButton button { border-radius: 5px; background-color: #FF4B4B; }
        </style>
    """, unsafe_allow_html=True)
    
    st.write(
        "This chatbot uses Hugging Face's models for text generation. "
        "You can customize the experience by selecting different models and datasets."
    )

def handle_chat_interaction(generator):
    """Handle chat messages and responses."""
    chat_container = st.container()
    with chat_container:
        for message in st.session_state['messages']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            try:
                response = generator(prompt, max_length=150, num_return_sequences=1)
                assistant_response = response[0]['generated_text']
            except Exception as e:
                st.error(f"Error generating response: {e}")
                return

        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        st.session_state['messages'].append({
            "role": "assistant",
            "content": assistant_response
        })

def main():
    """Main application function."""
    init_page_config()
    model_name = setup_sidebar()
    
    setup_chat_interface()
    
    generator = load_model(model_name)
    if generator:
        handle_chat_interaction(generator)
    else:
        st.error("Failed to load model. Please try selecting a different model.")

if __name__ == "__main__":
    main()