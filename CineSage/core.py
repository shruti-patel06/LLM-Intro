from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Initialize Model
model = init_chat_model("openai/gpt-oss-120b", model_provider="groq",temperature=0.9)

# Create Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
        """
        You are a movie information extraction assistant.

        Your task is to extract movie information from the
        paragraph provided by the user.

        STRICT RULES:
        1. Extract information ONLY from the given paragraph.
        2. Do not use your existing knowledge or assumptions.
        3. Do not invent, guess, or add missing information.
        4. If any information is not mentioned, write "Not specified".
        5. Preserve the original meaning of the paragraph.
        6. Do not introduce new facts or interpretations.
        7. Generate a quick summary using ONLY the information
           available in the paragraph.
        8. Keep the summary concise (2-3 sentences).

        Return the extracted information in the following format:

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
        "Extract movie information from the following paragraph:\n\n{movie_name}"
    )

])

# Input from User
movie = input("Enter movie Description:------>")

# Format Prompt
movie_prompt = prompt_template.invoke({"movie_name":movie})

# Get Response
response = model.invoke(movie_prompt)
print("\n Movie Details")
print(response.content)