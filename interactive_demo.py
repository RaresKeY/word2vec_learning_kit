import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from gensim.models import Word2Vec

# Configure Plotting Style
sns.set_theme(style="darkgrid")

MODEL_PATH = "models/word2vec_simple.model"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print("Please run 'python train.py' first.")
        sys.exit(1)
    
    print("Loading model... (this may take a second)")
    return Word2Vec.load(MODEL_PATH)

def plot_words(model, words):
    """
    Plots the given words in 2D space using t-SNE.
    Automatically adds similar words to the plot for context.
    """
    # 1. Expand the list with similar words for context
    expanded_words = set(words)
    for w in words:
        if w in model.wv:
            try:
                similar = model.wv.most_similar(w, topn=3)
                for sim_w, _ in similar:
                    expanded_words.add(sim_w)
            except: pass
            
    final_words = [w for w in expanded_words if w in model.wv]
    
    if len(final_words) < 2:
        print("Error: Need at least 2 valid words to plot.")
        return

    # 2. Get Vectors
    vectors = np.array([model.wv[w] for w in final_words])

    # 3. Reduce Dimensions (t-SNE)
    perplexity = min(30, len(final_words) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    vectors_2d = tsne.fit_transform(vectors)

    # 4. Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='red', edgecolors='k')

    for i, word in enumerate(final_words):
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), 
                     xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    title = f"Word Embeddings: {', '.join(words[:3])}..."
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    filename = "word_plot.png"
    plt.savefig(filename)
    print(f"\nPlot saved to '{filename}'")
    plt.close()

def main():
    clear_screen()
    model = load_model()
    
    print("\n--- INTERACTIVE MODE ---")
    print("Commands:")
    print("  word          -> Nearest neighbors (e.g., 'king')")
    print("  a - b + c     -> Analogy (e.g., 'paris - france + italy')")
    print("  plot: a,b,c   -> Save t-SNE plot")
    print("  exit          -> Quit")

    while True:
        try:
            command = input("\n>>> ").strip().lower()
        except KeyboardInterrupt:
            break
            
        if not command:
            continue
            
        if command in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
            
        # PLOT COMMAND
        if command.startswith("plot:"):
            raw_words = command.split(":", 1)[1]
            words = [w.strip() for w in raw_words.split(",")]
            plot_words(model, words)
            continue

        # VECTOR ARITHMETIC / LOOKUP
        # Allows: "king - man + woman", "science", "paris + germany", etc.
        try:
            positive = []
            negative = []
            current_sign = 1 # 1 = positive, -1 = negative
            
            # Normalize operators to ensure they are split correctly
            # e.g. "king-man" -> "king - man"
            normalized = command.replace("+", " + ").replace("-", " - ")
            tokens = normalized.split()
            
            for token in tokens:
                if token == "+":
                    current_sign = 1
                elif token == "-":
                    current_sign = -1
                else:
                    # It is a word
                    if current_sign == 1:
                        positive.append(token)
                    else:
                        negative.append(token)
            
            if not positive and not negative:
                print("Error: No valid words found in query.")
                continue
                
            # Formatting the output message
            query_parts = []
            if positive:
                query_parts.append(f"+({', '.join(positive)})")
            if negative:
                query_parts.append(f"-({', '.join(negative)})")
            print(f"Query: {' '.join(query_parts)}")

            # Perform the vector arithmetic
            results = model.wv.most_similar(positive=positive, negative=negative, topn=5)
            
            print("\nResults:")
            for w, score in results:
                print(f"  {w:<15} ({score:.3f})")
                
        except KeyError as e:
            print(f"Error: Word {e} not found in vocabulary.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
