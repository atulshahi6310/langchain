#!/usr/bin/env python
# coding: utf-8

# ### Useing ollama for the vector embedding useing the pretrain models

from langchain_community.embeddings import OllamaEmbeddings

# Create the embedding model
embed = OllamaEmbeddings(
        model="gemma:2b"

)

# Text to embed
documents = [
    "Artificial Intelligence is transforming the world.",
    "LangChain helps developers build powerful LLM applications.",
    "Ollama allows you to run language models locally.",
    "Embeddings convert text into numerical vectors.",
    "Retrieval-Augmented Generation improves the accuracy of AI responses.",
    "Python is one of the most popular programming languages.",
    "Machine learning models learn patterns from data.",
    "Vector databases enable efficient semantic search.",
    "Large Language Models can generate human-like text.",
    "Open-source AI tools are becoming increasingly powerful."
]

# Generate embeddings
embeddings = embed.embed_documents(documents)

# Print information
print("Number of embeddings:", len(embeddings))
print("Dimension of first embedding:", len(embeddings[0]))

print("\nFirst embedding (first 10 values):")
print(embeddings[0][:10])
