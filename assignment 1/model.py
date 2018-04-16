"""
Main module describing the IBM model 1 and 2 logic.
"""

# STD
from collections import defaultdict

# PROJECT



class Model:
    def __init__(self):
        self.cooc_counts = defaultdict(int)  # p(e_l, f_k): Co-occurrences between source and target language words
        self.source_counts = defaultdict(int)  # Count of words in the source language
        self.translation_probs = defaultdict(float)

    def train(self, data):
        pass


class Model1(Model):
    def __init__(self, alignment_prob):
        super().__init__()
        self.aignment_prob = alignment_prob


class Model2(Model):
    def __init__(self):
        super().__init__()
        self.alignment_counts = defaultdict(int)  # c(j|i, m, l)
        self.alignment_probs = defaultdict(float)
