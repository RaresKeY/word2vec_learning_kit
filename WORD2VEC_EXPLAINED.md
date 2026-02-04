# 🎓 Word2Vec Explained

This guide explains what is happening inside the `train.py` script in this learning kit.
While the "Masterclass" explains the deep math, this guide explains the **code you are actually running**.

---

## 1. The Goal
We want to take a stack of digital books (Sherlock Holmes, Frankenstein, etc.) and turn them into a "brain" that understands word meanings.

## 2. The Code: Step-by-Step

### Phase 1: Configuration (The Setup)
```python
BOOKS_URLS = [...]
VECTOR_SIZE = 100
WINDOW = 5
```
*   **What it does:** We list the books we want to read and set the "rules" for the AI's brain.
*   **Concept:**
    *   `VECTOR_SIZE`: How complex the thought for each word is. A size of 100 means "Dog" is described by 100 numbers.
    *   `WINDOW`: Context. How far left and right we look to understand a word.

### Phase 2: Data Preparation (The Download)
```python
def download_data():
    ...
    requests.get(url)
    ...
```
*   **What it does:** It goes to the internet (Project Gutenberg, GitHub), grabs text from various sources (books, Wikipedia articles), and saves them all into one big file: `datasets/corpus_simple.txt`.
*   **Why?** AI needs a lot of text to learn. We combine them so it learns from *all* the authors at once.

### Phase 3: The "Memory Friendly" Reader
```python
class MemoryFriendlyIterator:
    def __iter__(self):
        with open(self.filename) as f:
            for line in f:
                yield words
```
*   **The Problem:** If we had 1,000 books, loading them all into RAM would crash your computer.
*   **The Solution:** This piece of code is a **Streamer**. It reads **one line at a time** from the hard drive, hands it to the AI to learn, and then forgets it.
*   **Analogy:** Instead of memorizing a whole textbook instantly, you read it page by page.

### Phase 4: Training (The Magic)
```python
model = Word2Vec(
    sentences=sentences,
    vector_size=VECTOR_SIZE,
    ...
)
```
*   **What it does:** This one command launches the entire neural network training process using the **Gensim** library.
*   **Under the Hood:**
    1.  **Builds Vocab:** It scans the text first to find all unique words (e.g., "Sherlock", "elementary").
    2.  **Initializes Vectors:** It gives every word a random vector (random noise).
    3.  **Loops (Epochs):** It reads the text again and again (10 times).
    4.  **Adjusts:** Every time it sees "Sherlock" next to "Holmes", it nudges their vectors closer together.

### Phase 5: Saving the Brain
```python
model.save("models/word2vec_simple.model")
```
*   **What it does:** It takes the final list of numbers (vectors) and saves them to a file.
*   **Result:** You can now close the script, open `interactive_demo.py`, and load this file to play with the AI without re-training.

---

## 3. Key Takeaway
This script proves you don't need a supercomputer to train AI. By using **Streaming** (reading line-by-line) and **Efficient Libraries** (Gensim), you can train a smart model on a regular laptop in minutes.
