"""
Main module describing the IBM model 1 and 2 logic.
"""

# STD
import abc
from collections import defaultdict
import random
from itertools import product

# PROJECT
from corpus import ParallelCorpus


class Model:
    """
    Super class defining shared functions and variables for IBM models 1 and 2.
    """
    def __init__(self):
        self.cooc_counts = defaultdict(int)  # p(e_l, f_k): Co-occurrences between source and target language words
        self.source_counts = defaultdict(int)  # Count of words in the source language
        self.translation_probs = defaultdict(float)

    @abc.abstractmethod
    def train(self, data: ParallelCorpus, epochs=10):
        pass

    @staticmethod
    def add_null_token(sentence):
        return ["NULL"] + sentence


class Model1(Model):
    """
    Class for IBM model 1.
    """
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon  # Normalization constant

    def train(self, data: ParallelCorpus, epochs=10):
        self.translation_probs = defaultdict(lambda: 1 / data.source_vocab_size)  # Init uniformly

        for epoch in range(epochs):
            print("\nStarting epoch #{}...".format(epoch+1))
            self.cooc_counts = defaultdict(int)
            self.source_counts = defaultdict(int)

            for i, (source_sentence, target_sentence) in enumerate(data):
                print(
                    "\rProcessing sentence {}/{} ({:.2f} %)...".format(i+1, len(data), (i+1)/len(data)*100),
                    end="", flush=True
                )
                source_sentence = self.add_null_token(source_sentence)
                word_norms = defaultdict(float)

                # E-Step
                # Compute normalization
                # Implicitly uniform alignment probabilities as all alignments are considered equally
                for (source_token, target_token) in product(source_sentence, target_sentence):
                    word_norms[target_token] += self.translation_probs[(source_token, target_token)]

                # Collect counts
                for (source_token, target_token) in product(source_sentence, target_sentence):
                    pair = (source_token, target_token)
                    self.cooc_counts[pair] += self.translation_probs[pair] / word_norms[target_token]
                    self.source_counts[source_token] += self.translation_probs[pair] / word_norms[target_token]

                # M-Step
                # Estimate probabilities
                for (source_token, target_token) in product(source_sentence, target_sentence):
                    pair = (source_token, target_token)
                    self.translation_probs[pair] = self.cooc_counts[pair] / self.source_counts[source_token]

        print(self.translation_probs)



class Model2(Model):
    """
    Class for IBM model 2.
    """
    def __init__(self):
        super().__init__()
        self.alignment_counts = defaultdict(int)  # c(j|i, m, l)
        self.alignment_probs = defaultdict(float)


if __name__ == "__main__":
    corpus = ParallelCorpus(
        source_path="./data/training/hansards.36.2.f", target_path="./data/training/hansards.36.2.e"
    )
    model1 = Model1(epsilon=0.1)
    model1.train(corpus, epochs=4)
