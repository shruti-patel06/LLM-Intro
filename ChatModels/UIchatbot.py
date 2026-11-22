
import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Funny AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize model
@st.cache_resource
def get_model():
    return init_chat_model(
        "openai/gpt-oss-120b",
        model_provider="groq",
        temperature=0.9
    )

model = get_model()

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a funny AI assistant")
    ]

# UI Header
st.title("🤖 Funny AI Assistant")
st.caption("Chat with your funny AI assistant!")

st.divider()

# Display previous messages
for message in st.session_state.messages[1:]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# User input
prompt = st.chat_input("Write your message...")

if prompt:

    # Add human message
    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    # Display human message
    with st.chat_message("user"):
        st.write(prompt)

    # Get model response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(st.session_state.messages)

            st.write(response.content)

    # Add AI response
    st.session_state.messages.append(
        AIMessage(content=response.content)
    )
