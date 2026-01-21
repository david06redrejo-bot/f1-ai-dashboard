import spacy
from spacy import displacy
from collections import Counter, defaultdict
from pathlib import Path

# --- PART 1: Setup Logic (Copied from day2_ner_lm.py.py) ---
nlp = spacy.load("en_core_web_sm")

def setup_f1_ner(nlp_model):
    if not nlp_model.has_pipe("entity_ruler"):
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

class F1LanguageModel:
    def __init__(self):
        self.model = defaultdict(Counter)

    def train(self, sentences):
        for sentence in sentences:
            tokens = [token.text.lower() for token in nlp(sentence) if not token.is_punct]
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i+1]
                self.model[w1][w2] += 1

corpus = [
    "Box box for softs",
    "Box now for hards",
    "Box for new tires",
    "Hamilton is pitting now",
    "Verstappen is faster than Hamilton"
]

f1_lm = F1LanguageModel()
f1_lm.train(corpus)

# --- PART 2: Visualization Generation ---

# 1. NER Visualization
test_msg = "Hamilton needs to Box for Softs"
doc = nlp(test_msg)
ner_html = displacy.render(doc, style="ent", page=False)

# 2. Language Model Visualization (HTML Table)
lm_html_content = """
<div style="font-family: Arial, sans-serif; margin-top: 50px;">
    <h2>Bigram Language Model Frequencies</h2>
    <p>This table shows how often a word follows another word in our training corpus.</p>
    <table border="1" style="border-collapse: collapse; width: 50%;">
        <tr style="background-color: #f2f2f2;">
            <th style="padding: 8px;">Previous Word</th>
            <th style="padding: 8px;">Next Word</th>
            <th style="padding: 8px;">Count</th>
        </tr>
"""

for prev_word, follower_counts in f1_lm.model.items():
    for next_word, count in follower_counts.items():
        lm_html_content += f"""
        <tr>
            <td style="padding: 8px;">{prev_word}</td>
            <td style="padding: 8px;">{next_word}</td>
            <td style="padding: 8px;">{count}</td>
        </tr>
        """

lm_html_content += """
    </table>
</div>
"""

# Combine into a single HTML file
full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Day 2: NER & Language Model Visualization</title>
</head>
<body style="padding: 20px;">
    <h1>Day 2 Project Visualization</h1>
    
    <div style="margin-bottom: 50px;">
        <h2>Named Entity Recognition (NER)</h2>
        <p>Recognizing F1-specific entities (Drivers, Compounds, Actions).</p>
        {ner_html}
    </div>

    {lm_html_content}
</body>
</html>
"""

output_file = Path("day2_ner_lm.html")
output_file.write_text(full_html, encoding="utf-8")

print(f"Successfully created visualization at: {output_file.absolute()}")
