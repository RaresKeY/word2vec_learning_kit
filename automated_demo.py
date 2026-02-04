import os
from gensim.models import Word2Vec

def run_demo():
    model_path = "models/word2vec_simple.model"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found. Please run train.py first.")
        return

    print("🧠 Loading the Brain (Word2Vec Model)...")
    model = Word2Vec.load(model_path)
    wv = model.wv

    print("\n" + "="*50)
    print("   🌟 WORD2VEC AUTOMATED SHOWCASE 🌟")
    print("="*50)

    # 1. Similarity Check
    def check_similar(word):
        print(f"\n🔍 Searching for words similar to: '{word}'")
        try:
            results = wv.most_similar(word, topn=5)
            for i, (res, score) in enumerate(results):
                print(f"   {i+1}. {res:<15} (Match: {score:.2%})")
        except KeyError:
            print(f"   ⚠️ Word '{word}' not in vocabulary.")

    check_similar("sherlock")
    check_similar("science")
    check_similar("king")

    # 2. Analogy Check (The classic test)
    print("\n" + "-"*50)
    print("🧩 Analogy Test: King - Man + Woman = ?")
    try:
        results = wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)
        for i, (res, score) in enumerate(results):
            marker = "✅" if "queen" in res.lower() else "✨"
            print(f"   {marker} {res:<15} (Match: {score:.2%})")
    except KeyError as e:
        print(f"   ⚠️ Analogy failed: {e}")

    # 3. Odd One Out
    print("\n" + "-"*50)
    print("🧐 Odd One Out Test")
    
    test_lists = [
        ["apple", "banana", "car", "cherry"],
        ["sherlock", "watson", "holmes", "pizza"],
        ["biology", "physics", "chemistry", "london"]
    ]

    for words in test_lists:
        try:
            outlier = wv.doesnt_match(words)
            print(f"   List: {words}")
            print(f"   👉 The odd one is: '{outlier}'")
        except Exception as e:
            print(f"   ⚠️ Test failed for {words}: {e}")

    print("\n" + "="*50)
    print("   Showcase Complete! Run interactive_demo.py to explore more.")
    print("="*50)

if __name__ == "__main__":
    run_demo()
