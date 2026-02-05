# Training an NLP Model from Scratch

This guide walks you through the conceptual and practical steps of training a Word2Vec model, aligned with the code in `train.py`.

## Table of Contents
1.  [Gathering Data](#1-gathering-data)
2.  [Cleaning Data](#2-cleaning-data)
3.  [Vectorization & The "Black Box"](#3-vectorization--the-black-box)
    *   [How Skip-Gram Works](#how-skip-gram-works)
    *   [Dimensions](#dimensions)
4.  [Training (The Process)](#4-training-the-process)
    *   [Efficiency Note](#efficiency-note)
5.  [Technical Details (Deep Dive)](#5-technical-details-deep-dive)
    *   [Initial Vectorization (Random Start)](#1-initial-vectorization-random-start)
    *   [Calculating Loss (How wrong was I?)](#2-calculating-loss-how-wrong-was-i)
    *   [Backpropagation (Nudging Neurons)](#3-backpropagation-nudging-neurons)
    *   [Step-by-Step Training Loop](#4-step-by-step-training-loop)
6.  [Advanced Concepts & Modern Techniques](#6-advanced-concepts--modern-techniques)
    *   [N-Grams (Phrases)](#1-n-grams-phrases)
    *   [FastText (Subword Information)](#2-fasttext-subword-information)
    *   [Transformers (BERT/GPT)](#3-transformers--contextual-embeddings-bertgpt)

---

## 1. Gathering Data

The data we provide to the model determines its understanding of the world. "Garbage in, garbage out" applies heavily here. To build a robust model, we need clean text data from diverse sources:

*   **Formal Text (e.g., Wikipedia, Books):** Provides correct grammar and "grounding" in objective facts.
    *   *Conceptual Purpose:* Wiki data is dry and information-dense, grounding the model in "truth." It's excellent for encyclopedic knowledge but lacks emotional depth.
*   **Narrative Text (e.g., Novels):** Allows the model to see words in varied contexts, capturing thematic and emotional relationships.
    *   *Conceptual Purpose:* Novels provide the "EQ" (emotional intelligence) or intuition. Words are tied together in ephemeral, non-objective ways (e.g., "love" and "loss" appearing together), helping the model understand human sentiment.
*   **Informal Text (News/Chatter):** Helps the model pick up modern slang and evolving language usage (though our current kit focuses on the first two).

In `train.py`, we automate this by downloading a mix of classic literature (Sherlock Holmes, Frankenstein) and a subset of Wikipedia. This gives us a blend of vocabulary and structure.

## 2. Cleaning Data

Before the model can learn, we must simplify the text. Raw text contains noise that can confuse the model (e.g., "Word." and "word" might be treated as different things if we aren't careful).

In our `train.py` script, we use a `MemoryFriendlyIterator` to clean data on the fly as it is read from the disk. This is efficient and keeps our memory usage low.

**Our Cleaning Strategy:**
1.  **Lowercasing:** We convert "The" and "the" to the same token so the model learns they represent the same concept.
2.  **Noise Removal:** We use a Regular Expression (`re.sub(r'[^a-zA-Z\s]', '', line.lower())`) to keep only letters and spaces. Numbers and punctuation are removed to focus purely on word relationships.

*Note: In production environments, you might use more advanced tokenizers (like `spaCy` or `nltk`) to handle punctuation and special entities better, but for learning word embeddings, this simple approach is very effective.*

## 3. Vectorization & The "Black Box"

"Vectorizing" simply means converting words into lists of numbers ([vectors](#1-initial-vectorization-random-start)) positioned in a multi-dimensional space. The goal is to place semantically similar words close together in this space.

*   **The Concept:** Imagine a 2D graph. We want "King" and "Man" to be close, and "Queen" and "Woman" to be close.
*   **Doing Math on Meaning:** By placing words in space, we can perform arithmetic on their meanings.
    *   *Classic Analogy:* `King - Man + Woman = Queen`.
    *   If you take the vector for "King", subtract the "Man-ness" from it, and add "Woman-ness", the resulting position in space is closest to the vector for "Queen". This proves the model has learned the *concept* of gender, not just which words sit next to each other.
*   **Universal Approximation:** Conceptually, we are creating a mathematical function that embodies the spatial position (in our vocabulary) of the word. We don't need to visualize the 100 dimensions; we just need to trust that the math (Neural Networks) can approximate this "shape" of meaning, allowing us to compute relationships efficiently.

*   **The Math:** We use a **Skip-Gram** neural network approach (set via `sg=1` in the code).

### How Skip-Gram Works
Instead of predicting a word based on its context (fill-in-the-blank), Skip-Gram does the reverse. It takes a specific word (the "center" word) and tries to predict the words that likely appear around it (the "context").

*   **Example Sentence:** "The **king** owns these lands"
*   **Input:** "king"
*   **Target Output (Context):** "The", "owns", "these", "lands"

By repeatedly trying to predict context words from a center word, the model adjusts its internal numbers ([weights/neurons](#3-backpropagation-nudging-neurons)). Over time, these weights become the "vectors" that represent the word's meaning.

### Dimensions
In our code, we set `VECTOR_SIZE = 100`. This means every word is represented by a list of 100 numbers.
*   **Why 100?** It's a sweet spot for our dataset size.
*   **Why not 300?** Google uses 300 for massive models trained on billions of words. For our smaller library of books, 100 captures enough detail without overfitting or running too slowly.

## 4. Training (The Process)

Training happens in **Epochs**. One epoch is one full read-through of our entire library of books.

1.  **Streaming:** We don't load all books into RAM. The `MemoryFriendlyIterator` streams one line at a time.
2.  **Windowing:** The model looks at a "Window" of words. Our code sets `WINDOW = 5`, meaning it looks at 5 words before and 5 words after the center word.
3.  **Learning:**
    *   The model makes a prediction.
    *   It calculates the [Loss](#2-calculating-loss-how-wrong-was-i) (how wrong was I?).
    *   It updates the vectors slightly to reduce that error (see [Backpropagation](#3-backpropagation-nudging-neurons)).
4.  **Repeat:** We do this for `EPOCHS = 10`. Repeated exposure reinforces the patterns.

### Efficiency Note
The `gensim` library we use in `train.py` is highly optimized. It uses C code under the hood to perform these mathematical operations incredibly fast, utilizing multiple CPU cores (`workers=WORKERS`) to parallelize the job. This aligns perfectly with how CPUs handle standard number types (floats/ints), making training massively efficient.

Once training is complete, the model is saved to `models/word2vec_simple.model`. This file contains the "brain" – the learned vectors for every word in our vocabulary.

---

## 5. Technical Details (Deep Dive)

### 1. Initial Vectorization (Random Start)
How do we turn a word into numbers *before* we've learned anything? We cheat! We start with random noise.

*   **The Matrix:** We create a giant matrix (table) of size `Vocabulary Size x Vector Size`.
*   **Randomness:** We fill this table with tiny random numbers (e.g., between -0.01 and 0.01).
*   **Lookup:** Every unique word is assigned an index (e.g., "king" = 42). The vector for "king" is simply row 42 of this matrix.
*   **The Goal:** Training is simply the process of updating these random numbers until they represent meaning.

### 2. Calculating Loss (How wrong was I?)
To fix a mistake, we first need to measure it. In Word2Vec, "Loss" is calculated using **Negative Sampling** (an optimization of Softmax).

*   **The Problem:** Calculating the probability of "king" predicting "owns" against *every other word in the dictionary* (approx. 100,000 words) is too slow.
*   **The Solution (Negative Sampling):** We ask the model: "Is 'owns' the neighbor of 'king'?" (Yes). Then we pick 5 random words that *aren't* neighbors (e.g., "taco", "blue", "spaghetti") and ask: "Are these neighbors?" (No).
*   **The Math:** We use a Sigmoid function. We want the output to be `1` for the correct word and `0` for the random "negative" words. The difference between what we got (e.g., 0.3 for the real word) and what we wanted (1.0) is the **Loss**.

### 3. Backpropagation (Nudging Neurons)
Once we have the Loss, we need to change the vectors to do better next time. This is **Backpropagation** using **Gradient Descent**.

*   **The Gradient:** Imagine standing on a hill (high error) and feeling with your feet which way is "down" (lower error). The gradient is that direction.
*   **The Update:** We take the current vector numbers and subtract a tiny fraction of the gradient.
    *   `New Weight = Old Weight - (Learning Rate * Gradient)`
*   **The Result:** The vector for "king" moves slightly closer (mathematically) to "owns" and slightly further away from "taco".

### 4. Step-by-Step Training Loop
Here is exactly what happens for a single training step with the sentence "The **king** owns...":

1.  **Lookup:** The model grabs the current vector for the input word "king" (Row 42).
2.  **Projection:** It multiplies this vector by a weight matrix to get scores for potential context words.
3.  **Prediction:** It applies the Sigmoid function to these scores to get probabilities.
4.  **Error Calculation:** It compares the probability of the target word ("owns") to 1.0, and the negative samples ("taco") to 0.0.
5.  **Backprop:** It calculates the gradient (the "slope" of the error).
6.  **Update:** It tweaks the numbers in the "king" vector and the "owns" vector to reduce the error.
7.  **Next Word:** It moves to the next word in the sentence and repeats.

---

## 6. Advanced Concepts & Modern Techniques

While we start with standard Word2Vec, modern NLP has evolved. Here are key techniques used in production:

### 1. N-Grams (Phrases)
"New York" implies a city, but "New" and "York" separately mean "recent" and "a city in England".
*   **Technique:** We can pre-process text to find words that statistically appear together more often than chance.
*   **Result:** We fuse them into a single token: `new_york`. The model then learns a specific vector for this unique concept, separate from "new" or "york".

### 2. FastText (Subword Information)
What happens if the model sees the word "unforgettableness" but has never trained on it? Word2Vec crashes (or ignores it).
*   **Technique (FastText):** Developed by Facebook, this treats words as bags of character n-grams.
    *   "apple" becomes `<ap`, `app`, `ppl`, `ple`, `le>`.
*   **Benefit:** The model can construct a meaning for "unforgettableness" by combining the vectors for "un-", "forget", "-able", and "-ness". It understands words it has never seen before!

### 3. Transformers & Contextual Embeddings (BERT/GPT)
Word2Vec has a flaw: "Bank" has the same vector in "river bank" and "bank deposit".
*   **Technique (Transformers):** Models like BERT don't have static vectors. They generate a vector *on the fly* based on the entire sentence.
*   **Result:** The vector for "bank" changes completely depending on the surrounding words. This is the foundation of modern LLMs (Large Language Models) like the one helping you right now.
