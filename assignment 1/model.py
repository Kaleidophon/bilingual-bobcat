"""
Main module describing the IBM model 1 and 2 logic.
"""

# STD
import abc
from collections import defaultdict
import time
from itertools import product

# EXT
import numpy as np

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

    def train(self, data: ParallelCorpus, epochs=10, verbosity=1):
        self.translation_probs = defaultdict(lambda: 1 / data.source_vocab_size)  # Init uniformly
        log_likelihoods = defaultdict(float)
        num_sentences = len(data)
        print_interval = int(num_sentences / 10000)

        for epoch in range(epochs):
            start = time.time()
            self.cooc_counts = defaultdict(int)
            self.source_counts = defaultdict(int)

            if verbosity > 0:
                print("\nStarting epoch #{}...".format(epoch+1))

            for i, (source_sentence, target_sentence) in enumerate(data):
                source_sentence = self.add_null_token(source_sentence)
                word_norms = defaultdict(float)
                sentence_log_likelihood = 0

                if verbosity > 0 and (i+1) % print_interval == 0:
                    print(
                        "\rProcessing sentence {}/{} ({:.2f} %)...".format(i+1, len(data), (i+1)/len(data)*100),
                        end="", flush=True
                    )

                # E-Step
                # Compute normalization
                # Implicitly uniform alignment probabilities as all alignments are considered equally
                for (source_token, target_token) in product(source_sentence, target_sentence):
                    word_norms[target_token] += self.translation_probs[(source_token, target_token)]

                # Collect counts
                for source_token in source_sentence:
                    source_token_probs = 0  # Sum of pi(f_j|e_i) over all i
                    for target_token in target_sentence:
                        pair = (source_token, target_token)
                        self.cooc_counts[pair] += self.translation_probs[pair] / word_norms[target_token]
                        self.source_counts[source_token] += self.translation_probs[pair] / word_norms[target_token]
                        source_token_probs += self.translation_probs[(source_token, target_token)]

                    sentence_log_likelihood += np.log(source_token_probs)

                # Normalization of sentence log-likelihood by sentence lengths
                sentence_log_likelihood += np.log(self.epsilon) - len(target_sentence) * np.log(len(source_sentence))
                log_likelihoods[epoch] += sentence_log_likelihood

            # M-Step
            # Estimate probabilities
            for (source_token, target_token) in self.translation_probs.keys():
                pair = (source_token, target_token)
                self.translation_probs[pair] = self.cooc_counts[pair] / self.source_counts[source_token]

            end = time.time()
            duration = end - start
            m, s = divmod(duration, 60)
            if verbosity > 0:
                print(
                    "\rLog-likelihood for epoch #{}: {:.4f}\nEpoch #{} took {} minute(s) and {:.2f} second(s).\n".format(
                        epoch+1, log_likelihoods[epoch], epoch+1, m, s
                    ),
                    end="", flush=True
                )

        if verbosity > 0:
            sorted_translation_probs = sorted(self.translation_probs.items(), key=lambda tpl: tpl[1], reverse=True)
            for (source_token, target_token), prob in sorted_translation_probs[:50]:
                print("{} -> {}: {:.6f}".format(source_token, target_token, prob))




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
