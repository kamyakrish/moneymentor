import openai
import pandas as pd
import os

# Ensure you have set up your OpenAI API key
  # Alternatively, set it manually

# Load the cleaned dataset
merged_df = pd.read_csv("financial_glossary.csv")  # Use the correct file path

# Function to generate embeddings using OpenAI's latest API
def generate_embeddings(text_list):
    response = openai.embeddings.create(
        input=text_list,
        model="text-embedding-ada-002"
    )
    return [item.embedding for item in response.data]

# Generate embeddings for the definitions
merged_df["embedding"] = generate_embeddings(merged_df["definition"].tolist())

# Save the dataset with embeddings
merged_df.to_csv("financial_glossary_with_embeddings.csv", index=False)

print("Embeddings successfully generated and saved!")
