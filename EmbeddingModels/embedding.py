from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
texts = [
        "Hey, My name is Shruti and I am your AI teacher",
        "I love creating real-world solutions",
        "I enjoy coding as it is in blood"
]
vector = embedding.embed_documents(texts)

print(vector)