import spacy
from spacy import displacy
from pathlib import Path

# 1. Load the model
nlp = spacy.load("en_core_web_sm")

# 2. Test data
radio_messages = [
    "Hamilton is complaining about the tires",
    "Box box for softs",
    "Verstappen has no grip",
    "Engine is overheating rapidly"
]

# 3. Create the 'Doc' objects (the trees)
docs = [nlp(msg) for msg in radio_messages]

# 4. Generate the visualization HTML
# style="dep" stands for 'dependency'
html = displacy.render(docs, style="dep", page=True)

# 5. Save to a file
output_file = Path("syntax_trees.html")
output_file.write_text(html, encoding="utf-8")

print(f"Successfully created syntax trees at: {output_file.absolute()}")
print("Please open this file in your browser to see the trees.")
