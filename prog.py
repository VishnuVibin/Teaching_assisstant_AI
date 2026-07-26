from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Take input from the user
sentence = input("Enter a sentence: ")

# Generate the embedding
embedding = model.encode(sentence)

# Print the embedding
print("\nEmbedding:")
print(embedding)

# Print some additional information
print("\nLength of embedding:", len(embedding))