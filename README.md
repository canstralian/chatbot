# 💬 Chatbot

[![CI](https://github.com/canstralian/chatbot/actions/workflows/tests.yml/badge.svg)](https://github.com/canstralian/chatbot/actions/workflows/tests.yml)
[![Linting](https://github.com/canstralian/chatbot/actions/workflows/linting.yml/badge.svg)](https://github.com/canstralian/chatbot/actions/workflows/linting.yml)
[![Deployment](https://github.com/canstralian/chatbot/actions/workflows/cd.yml/badge.svg)](https://github.com/canstralian/chatbot/actions/workflows/cd.yml)
[![Security](https://github.com/canstralian/chatbot/actions/workflows/security.yml/badge.svg)](https://github.com/canstralian/chatbot/actions/workflows/security.yml)

A simple Streamlit app that demonstrates a chatbot using Hugging Face's text generation models.

### Features
- Choose from different Hugging Face models (gpt2, distilgpt2, EleutherAI/gpt-neo-125M)
- Interactive chat interface
- Message history persistence

### How to use
1. Select a model from the dropdown menu
2. Type your message in the chat input
3. The chatbot will generate and display a response

### Getting Started
1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   streamlit run streamlit_app.py
   ```

The app will be accessible through your web browser.
