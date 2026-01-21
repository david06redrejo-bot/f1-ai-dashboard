import spacy

# Load the medium model which contains 20k unique vectors
# Each vector has 300 dimensions (300 numbers representing the word)
nlp = spacy.load("en_core_web_md")

def analyze_embeddings():
    # Define F1 keywords
    word1 = "Hamilton"
    word2 = "Mark"
    word3 = "Tire"
    word4 = "Engine"

    # Process words to get their vector representations
    token1 = nlp(word1)
    token2 = nlp(word2)
    token3 = nlp(word3)
    token4 = nlp(word4)

    # Calculate Cosine Similarity (Result is between 0 and 1)
    sim_drivers = token1.similarity(token2)
    sim_parts = token3.similarity(token4)
    sim_mix = token1.similarity(token3)

    print("--- Similarity Analysis ---")
    # Using the requested print syntax
    print("Given word: " + word1 + " and " + word2 + " -> Similarity score: " + str(sim_drivers))
    print("Given word: " + word3 + " and " + word4 + " -> Similarity score: " + str(sim_parts))
    print("Given word: " + word1 + " and " + word3 + " -> Similarity score: " + str(sim_mix))

    # Accessing the raw vector (first 5 dimensions only for brevity)
    print("\n--- Vector Preview (Dimensions) ---")
    print("Word: " + word1 + " -> First 5 vector dims: " + str(token1.vector[:5]))

# Running the simulation
if __name__ == "__main__":
    analyze_embeddings()
    
    # Logic for your app: Prediction based on proximity
    # If the user types 'Motor', the embedding helps the AI know it's related to 'Engine'
    query = "Motor"
    base = "Engine"
    if nlp(query).similarity(nlp(base)) > 0.7:
        prediction = "Technical Issue Detected"
        current_word = query
        print("\n--- Prediction Logic ---")
        print("Given word: " + current_word + " -> Predicted next word: " + prediction)