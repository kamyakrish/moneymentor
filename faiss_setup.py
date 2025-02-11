import faiss
import numpy as np
import pandas as pd
import pickle

# Load embeddings
df = pd.read_csv("financial_glossary_with_embeddings.csv")
df["embedding"] = df["embedding"].apply(eval)  # Convert string to list

# Convert embeddings to numpy array
embeddings = np.array(df["embedding"].tolist()).astype("float32")

# Initialize FAISS index
index = faiss.IndexFlatL2(1536)  # OpenAI embeddings are 1536-d
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "faiss_index.idx")

# Save metadata
df[["term", "definition"]].to_pickle("faiss_metadata.pkl")

print("✅ FAISS setup complete!")
