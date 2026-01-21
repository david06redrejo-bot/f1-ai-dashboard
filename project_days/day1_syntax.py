import spacy

# 1. Load the brain (pre-trained model)
# This loads the neural network weights to tag words.
nlp = spacy.load("en_core_web_sm")

# Test data (simulated Radio Transcripts)
radio_messages = [
    "Hamilton is complaining about the tires",
    "Box box for softs",
    "Verstappen has no grip",
    "Engine is overheating rapidly"
]

def analyze_syntax(text):
    # 2. The Magic Pipeline
    # When calling nlp(text), spaCy tokenizes and predicts syntactic relationships.
    doc = nlp(text)
    
    print("\n--- Analyzing: " + text + " ---")
    
    # 3. Data Extraction
    # We iterate over each 'token' (word) in the processed document.
    for token in doc:
        # token.text: The literal word.
        # token.dep_: The syntactic dependency (nsubj, ROOT, dobj...).
        # token.head.text: The 'parent' word it depends on in the tree.
        # token.pos_: Part-of-Speech (Verb, Noun, Adjective).
        
        print(token.text + " | Role: " + token.dep_ + " | Parent: " + token.head.text + " | Type: " + token.pos_)

# Execution
for msg in radio_messages:
    analyze_syntax(msg)