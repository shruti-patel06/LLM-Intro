
import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬",
    layout="centered"
)

# Initialize Model
@st.cache_resource
def get_model():
    return init_chat_model(
        "openai/gpt-oss-120b",
        model_provider="groq",
        temperature=0
    )

model = get_model()

# Chat Prompt Template
movie_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a movie information extraction assistant.

        Extract information ONLY from the provided paragraph.

        Rules:
        - Never add information from your existing knowledge.
        - Never guess or assume missing information.
        - Use "Not specified" for missing information.
        - Preserve the original meaning.
        - Generate a 2-3 sentence summary using only
          information from the paragraph.

        Output format:

        Movie Name:
        Release Year:
        IMDb Rating:
        Director:
        Starring:
        Genre:
        Runtime:
        Language:

        Movie Description:
        Quick Summary:
        """
    ),
    (
        "human",
        "{movie_description}"
    )
])

# Streamlit UI
st.title("🎬 Movie Information Extractor")

st.write(
    "Enter a movie description to extract structured "
    "information and generate a quick summary."
)

st.divider()

# User Input
movie_description = st.text_area(
    "Enter Movie Description",
    placeholder="Paste your movie paragraph here...",
    height=200
)

# Extract Button
if st.button("Extract Movie Information", use_container_width=True):

    if movie_description.strip():

        with st.spinner("Extracting movie information..."):

            # Format Prompt
            formatted_prompt = movie_prompt.invoke({
                "movie_description": movie_description
            })

            # Generate Response
            response = model.invoke(formatted_prompt)

        st.subheader("📽️ Extracted Movie Information")

        st.markdown(response.content)

    else:
        st.warning("Please enter a movie description first.")
