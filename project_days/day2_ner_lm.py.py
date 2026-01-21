import spacy
from spacy.language import Language
from collections import Counter, defaultdict

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# --- PART 1: Sequence Labeling (Custom NER) ---
def setup_f1_ner(nlp_model):
    # Add EntityRuler to the pipeline to identify F1-specific terms
    # We add this 'entity_ruler' BEFORE the standard 'ner' component.
    # This ensures our custom patterns (like 'Hamilton' -> DRIVER) are applied FIRST.
    # Rules take precedence: if we say 'Softs' is a COMPOUND, the statistical model won't overwrite it.
    ruler = nlp_model.add_pipe("entity_ruler", before="ner")
    
    patterns = [
        {"label": "DRIVER", "pattern": "Hamilton"},
        {"label": "DRIVER", "pattern": "Verstappen"},
        {"label": "DRIVER", "pattern": "Leclerc"},
        {"label": "COMPOUND", "pattern": "Softs"},
        {"label": "COMPOUND", "pattern": "Hards"},
        {"label": "COMPOUND", "pattern": "Inters"},
        {"label": "ACTION", "pattern": "Box"},
        {"label": "ACTION", "pattern": "Pit"}
    ]
    ruler.add_patterns(patterns)
    return nlp_model

nlp = setup_f1_ner(nlp)

# --- PART 2: Language Modeling (Simple Bigram Model) ---
class F1LanguageModel:
    def __init__(self):
        # Dictionary to store word frequencies: {previous_word: {current_word: count}}
        self.model = defaultdict(Counter)

    def train(self, sentences):
        for sentence in sentences:
            # Process the sentence with spaCy to get tokens, 
            # here token.is_punct filters out punctuation 
            # (e.g., 'Box, box!' -> ['box', 'box'], without the ',' and '!'), 
            # returns a boolean for each token, depending on whether it is punctuation or not.
            tokens = [token.text.lower() for token in nlp(sentence) if not token.is_punct]
            # Create bigrams: (w1, w2), (w2, w3)...
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i+1]
                self.model[w1][w2] += 1

    def predict_next(self, word):
        word = word.lower()
        if word not in self.model:
            return "Unknown"
        # .most_common(n) returns a list of the 'n' most frequent (element, count) tuples.
        # most_common(1) returns something like: [('for', 3)]
        # [0] access the first tuple: ('for', 3)
        # [0] access the first element of that tuple: 'for'
        return self.model[word].most_common(1)[0][0]

# --- PART 3: Execution ---

# Training data for our mini Language Model
corpus = [
    "Box box for softs",
    "Box now for hards",
    "Box for new tires",
    "Hamilton is pitting now",
    "Verstappen is faster than Hamilton"
]

# Initialize and train LM
f1_lm = F1LanguageModel()
f1_lm.train(corpus)

# Test Document
test_msg = "Hamilton needs to Box for Softs"
doc = nlp(test_msg)

print(f"--- NER Analysis: '{test_msg}' ---")
for ent in doc.ents:
    print(f"Entity: {ent.text:<12} | Label: {ent.label_}")

print("\n--- Language Model Prediction ---")
current_word = "Box"
prediction = f1_lm.predict_next(current_word)
print(f"Given word: '{current_word}' -> Predicted next word: '{prediction}'")