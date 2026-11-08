import os
import sqlite3
import faiss
import numpy as np
from variables import *

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS mapping (
        faiss_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    )
''')
conn.commit()

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    
def register_face(name, embedding_vector):
    global index
    vector = np.array(embedding_vector, dtype=np.float32).reshape(1, -1)
    
    faiss_id = index.ntotal
    
    index.add(vector)
    cursor.execute("INSERT INTO mapping (faiss_id, name) VALUES (?, ?)", (faiss_id, name))
    conn.commit()
    
    faiss.write_index(index, INDEX_FILE)
    print(f"Successfully registered {name} with ID {faiss_id}")
    
def recognize_face(current_embedding, threshold = 0.6):
    vector = np.array(current_embedding, dtype=np.float32).reshape(1, -1)
    
    distances, indices = index.search(vector, k = 1)
    
    best_match_id = indices[0][0]
    distance = distances[0][0]
    
    if best_match_id == -1 or distance > threshold:
        return "Unknown", distance
    
    cursor.execute("SELECT name FROM mapping WHERE faiss_id = ?", (int(best_match_id),))
    row = cursor.fetchone()
    
    if row:
        return row[0], distance
    return "Unknown", distance