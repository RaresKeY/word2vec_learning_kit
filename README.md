# Simplified Word2Vec Learning Kit

This simplified project is designed to teach the fundamentals of Word2Vec using **Gensim**, without the complexity of manual PyTorch tensor operations.

## 1. How it works
We use **Gensim**, a library optimized for "Topic Modelling for Humans". It implements the Word2Vec algorithms (Skip-Gram and CBOW) in highly optimized C, making it fast and accurate.

## 2. Project Structure

*   `train.py`: The "Teacher".
    *   Downloads a small but diverse set of texts (classic books, Wikipedia articles, etc.).
    *   Reads them line-by-line (streaming).
    *   Trains a neural network to predict words from context.
    *   Saves the "Brain" (Model) to disk.
*   `interactive_demo.py`: The "Playground".
    *   Loads the trained brain.
    *   Lets you ask it questions via a simple menu.

## 3. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
This will take about 1-2 minutes.
```bash
python train.py
```

### Step 3: Verify with Automated Demo
Run a quick health-check to see if the model learned anything useful.
```bash
python automated_demo.py
```

### Step 4: Play with the AI
Explore the model interactively.
```bash
python interactive_demo.py
```

## 4. Concepts to Explore

### Similarity
Ask for similar words to `sherlock`. You might see `holmes`, `watson`, or `detective`. The model learned these are related because they appear in the same sentences.

### Analogies
Try the classic:
*   A: `man`
*   B: `king`
*   C: `woman`
*   **Result:** `queen` (Hopefully!)

*Math:* `Vector(King) - Vector(Man) + Vector(Woman) = Vector(Queen)`

### Odd One Out
Give it a list: `apple banana car cherry`.
It should pick `car` because fruits appear in similar contexts (eating, growing), while cars appear in different contexts (driving, roads).
