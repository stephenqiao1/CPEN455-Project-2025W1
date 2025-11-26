from model.tokenizer import Tokenizer
import os
from dotenv import load_dotenv

load_dotenv()
checkpoint = os.getenv("MODEL_CHECKPOINT")
cache_dir = os.getenv("MODEL_CACHE_DIR")
tok = Tokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)

# List of potential candidates to test
candidates = [
    # Case variations of Spam
    "spam", "Spam", "SPAM", 
    # Case variations of Ham/Safe
    "ham", "Ham", "HAM",
    "safe", "Safe", "SAFE",
    "clean", "Clean", "CLEAN",
    "legit", "Legit", "LEGITIMATE",
    # Binary concepts
    "yes", "Yes", "YES",
    "no", "No", "NO",
    "true", "True", "TRUE",
    "false", "False", "FALSE",
    "bad", "Bad", "BAD",
    "good", "Good", "GOOD",
    "junk", "Junk", "JUNK",
    "malicious", "threat", "danger"
]

print(f"{'WORD':<15} | {'TOKENS':<10} | {'STATUS'}")
print("-" * 40)

good_candidates = []

for word in candidates:
    # Encode
    ids = tok.encode(word)
    is_single = len(ids) == 1
    
    status = "✅ SINGLE" if is_single else f"❌ {len(ids)} parts"
    print(f"{word:<15} | {str(ids):<10} | {status}")
    
    if is_single:
        good_candidates.append(word)

print("-" * 40)
print("RECOMMENDED PAIRS (Pick one pair where BOTH are Single Tokens):")

# logic to suggest pairs
if "spam" in good_candidates and "ham" in good_candidates:
    print("1. spam / ham  (Classic)")
if "bad" in good_candidates and "good" in good_candidates:
    print("2. bad / good  (Simple Semantic)")
if "true" in good_candidates and "false" in good_candidates:
    print("3. true / false (Strongest Priors)")