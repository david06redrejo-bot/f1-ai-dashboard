import spacy
import numpy as np
from pathlib import Path

# --- PART 1: Setup Logic (Based on day3_embeddings.py) ---
# Load the medium model which contains 20k unique vectors
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Warning: 'en_core_web_md' not found. Falling back to 'en_core_web_sm'. Results may be poor.")
    nlp = spacy.load("en_core_web_sm")

def get_similarity_matrix(keywords):
    tokens = [nlp(w) for w in keywords]
    n = len(keywords)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            matrix[i][j] = tokens[i].similarity(tokens[j])
            
    return matrix

# Define keywords for visualization
keywords = ["Hamilton", "Verstappen", "Tire", "Engine", "Softs", "Mercedes", "Ferrari", "Fast", "Slow"]

sim_matrix = get_similarity_matrix(keywords)

# --- PART 2: Visualization Generation (Heatmap) ---

def color_from_value(val):
    # Blue (low similarity) to Red (high similarity)
    # val is 0 to 1
    
    # 0 -> 255, 255, 255 (White) is clearer? Or Blue?
    # Let's do White to Green for simplicity and standard heatmap look
    # or Red (1.0) to White (0.0)
    
    # Let's do a simple R,G,B interpolation
    # Low (0.0): White (255, 255, 255)
    # High (1.0): Dark Green (0, 100, 0)
    
    r = int(255 * (1 - val))
    g = int(255 * (1 - val/2)) # slightly greener
    b = int(255 * (1 - val))
    
    # Better Scheme:
    # > 0.7: Green
    # > 0.4: Yellow
    # < 0.4: Red/Gray
    
    if val > 0.7:
        return f"rgba(0, 255, 0, {val*0.6})" # Green with opacity
    elif val > 0.4:
        return f"rgba(255, 255, 0, {val*0.6})" # Yellow
    else:
        return f"rgba(200, 200, 200, 0.3)" # Gray
        
    return f"rgb({r},{g},{b})"

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Day 3: Word Embeddings Heatmap</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        table { border-collapse: collapse; }
        td, th { padding: 10px; border: 1px solid #ddd; text-align: center; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Day 3 Project Visualization: Embeddings Heatmap</h1>
    <p>This heatmap shows the Cosine Similarity between F1-related terms. 
       <br>Green = High Similarity (Close meaning/context)
       <br>Yellow = Medium Similarity
       <br>Gray = Low Similarity
    </p>
    
    <table>
        <tr>
            <th></th>
"""

# Header Row
for kw in keywords:
    html_content += f"<th>{kw}</th>"
html_content += "</tr>"

# Data Rows
for i, row_kw in enumerate(keywords):
    html_content += f"<tr><th>{row_kw}</th>"
    for j, col_kw in enumerate(keywords):
        score = sim_matrix[i][j]
        color = color_from_value(score)
        html_content += f'<td style="background-color: {color};">{score:.2f}</td>'
    html_content += "</tr>"

html_content += """
    </table>
</body>
</html>
"""

output_file = Path("day3_embeddings.html")
output_file.write_text(html_content, encoding="utf-8")

print(f"Successfully created visualization at: {output_file.absolute()}")
