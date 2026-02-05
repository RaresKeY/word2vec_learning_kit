import os
import requests
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import logging
from pathlib import Path

# Setup Logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ========================================== 
# 1. CONFIGURATION
# ========================================== 
BOOKS_URLS = [
    # A mix of classic literature for good variety
    "https://www.gutenberg.org/files/1661/1661-0.txt",    # Sherlock Holmes
    "https://www.gutenberg.org/files/1342/1342-0.txt",    # Pride and Prejudice
    "https://www.gutenberg.org/files/84/84-0.txt",        # Frankenstein
    "https://www.gutenberg.org/files/11/11-0.txt",        # Alice in Wonderland
    "https://www.gutenberg.org/files/2012/2012-0.txt",    # Origin of Species (Science)
    "https://www.gutenberg.org/cache/epub/1497/pg1497.txt", # The Republic (Philosophy)
    
    # WikiText-2 (High Quality Wikipedia subset - ~2 million words)
    "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
]

DATA_FILE = "datasets/corpus_simple.txt"
MODEL_PATH = "models/word2vec_simple.model"

Path(DATA_FILE).parent.mkdir(parents=True)
Path(MODEL_PATH).parent.mkdir(parents=True)

# ==========================================
# 2. HYPERPARAMETERS (The "Knobs" of the Brain)
# ==========================================
# VECTOR_SIZE: How detailed the mental image of a word is.
# 100 is a good balance for small datasets. Google uses 300.
VECTOR_SIZE = 100

# WINDOW: How far the AI looks to the left and right of a word.
# 5 means it looks at 5 neighbors. "The quick brown [FOX] jumps over the..."
# Larger window = learns topics. Smaller window = learns grammar.
WINDOW = 5

# MIN_COUNT: The "Ignore Rare Words" threshold.
# If a word appears less than 5 times, we ignore it.
# This filters out typos and names that don't teach us general meanings.
MIN_COUNT = 5

WORKERS = multiprocessing.cpu_count() # Use all CPU cores
EPOCHS = 10 # How many times to read the whole library (Practice makes perfect!)

# ========================================== 
# 2. DATA PREPARATION
# ========================================== 
class MemoryFriendlyIterator:
    """Streams lines from disk to save RAM."""
    def __init__(self, filename):
        self.filename = filename
        
    def __iter__(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                # Simple cleaning
                import re
                clean_line = re.sub(r'[^a-zA-Z\s]', '', line.lower())
                words = clean_line.split()
                if len(words) > 1:
                    yield words

def download_data():
    if os.path.exists(DATA_FILE):
        print(f"Using existing data file: {DATA_FILE}")
        return

    print("Downloading data...")
    full_text = ""
    for url in BOOKS_URLS:
        try:
            print(f"Fetching {url}...")
            response = requests.get(url)
            response.encoding = 'utf-8'
            text = response.text
            # Simple Gutenberg cleaning
            start = text.find("*** START OF")
            end = text.find("*** END OF")
            if start != -1 and end != -1:
                text = text[start:end]
            full_text += text + "\n"
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"Saved corpus to {DATA_FILE}")

# ========================================== 
# 3. TRAINING
# ========================================== 
class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    def on_epoch_begin(self, model):
        print(f"Epoch {self.epoch} starting...")
    def on_epoch_end(self, model):
        print(f"Epoch {self.epoch} finished.")
        self.epoch += 1

def main():
    download_data()
    
    print(f"Training Word2Vec (Workers: {WORKERS})...")
    sentences = MemoryFriendlyIterator(DATA_FILE)
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        epochs=EPOCHS,
        sg=1, # Skip-Gram
        callbacks=[EpochLogger()]
    )
    
    model.save(MODEL_PATH)
    print(f"Success! Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
