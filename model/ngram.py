import math
from collections import defaultdict


class NgramExpert:
    """
    A simple Unigram Language Model that acts as the 'Statistical Expert'.
    It calculates the probability of a text belonging to a class based on word frequencies.
    """

    def __init__(self, smoothing=1.0):
        self.counts = defaultdict(int)
        self.total_count = 0
        self.vocab = set()
        self.smoothing = smoothing

    def train(self, texts):
        """
        Train the model on a list of strings.
        """
        for text in texts:
            # Simple tokenization by splitting on whitespace
            tokens = str(text).lower().split()
            for token in tokens:
                self.counts[token] += 1
                self.total_count += 1
                self.vocab.add(token)

    def get_log_prob(self, text):
        """
        Calculate the log-probability of a text: sum(log(P(word)))
        """
        tokens = str(text).lower().split()
        if not tokens:
            # Empty text has zero probability
            return 0.0
        
        log_prob = 0.0
        vocab_size = len(self.vocab)
        
        # Handle case where model hasn't been trained or has no vocabulary
        if self.total_count == 0:
            # If no training data, use uniform distribution over a default vocabulary size
            # This prevents division by zero
            default_vocab_size = 10000  # Reasonable default for unseen tokens
            for token in tokens:
                prob = self.smoothing / (self.smoothing * default_vocab_size)
                log_prob += math.log(prob)
            return log_prob
        
        # Normal case: Laplace Smoothing: (Count + alpha) / (Total + alpha * VocabSize)
        # Ensure denominator is never zero
        denominator = self.total_count + self.smoothing * max(vocab_size, 1)
        
        for token in tokens:
            count = self.counts.get(token, 0)
            prob = (count + self.smoothing) / denominator
            log_prob += math.log(prob)
            
        return log_prob

