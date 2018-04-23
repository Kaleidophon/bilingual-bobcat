"""
Main module describing the IBM model 1 and 2 logic.
"""

# STD
import abc
from collections import defaultdict
import time
from itertools import product
import pickle
import random

# EXT
import numpy as np

# PROJECT
from aer import read_naacl_alignments, AERSufficientStatistics
from corpus import ParallelCorpus

# TODO
# - IBM2 log-likelihood falls after some point instead of increasing


class AlignmentProbDict:
    """
    Class to store alignment probabilities which can take an argument when returning a default value (which defaultdict
    can't).
    """
    def __init__(self):
        self.core = defaultdict(float)

    def __getitem__(self, item):
        if item in self.core:
            return self.core[item]
        else:
            _, length_target, _, _ = item
            self.core[item] = 1 / (length_target + 1)  # Initialize alignment probability uniformly
            return self.core[item]

    def __setitem__(self, key, value):
        self.core[key] = value

    def __len__(self):
        return len(self.core)

    def keys(self):
        return self.core.keys()

    def items(self):
        return self.core.items()

    def __repr__(self):
        return self.core.__repr__()

    def __str__(self):
        return self.core.__str__()


class Model:
    """
    Super class defining shared functions and variables for IBM models 1 and 2.
    """
    def __init__(self, eval_alignment_path=None, eval_corpus=None):
        self.eval_alignment_path = eval_alignment_path
        self.gold_standard = None
        self.eval_corpus = eval_corpus
        if self.eval_alignment_path is not None:
            self.gold_standard = read_naacl_alignments(self.eval_alignment_path)

        self.cooc_counts = defaultdict(float)  # p(e_l, f_k): Co-occurrences between source and target language words
        self.source_counts = defaultdict(float)  # Count of words in the source language
        self.translation_probs = defaultdict(float)

    def save(self, path):
        with open(path, "wb") as file:
            self.translation_probs = dict(self.translation_probs)  # You can't pickle lambda functions
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)

    def init_translation_probabilities(self, mode, data, given_probs):
        assert mode in ("uniform", "random", "continue"), "Invalid initialization mode {}".format(mode)

        # Initialize uniformly
        if mode == "uniform":
            self.translation_probs = defaultdict(lambda: 1 / data.source_vocab_size)

        # Initialize randomly
        elif mode == "random":
            self.translation_probs = defaultdict(lambda: random.random())

        # Initialize with already trained probabilities
        elif mode == "continue":
            assert given_probs is not None
            self.translation_probs = given_probs

    def train(self, data: ParallelCorpus, epochs=10, initialization="uniform", verbosity=1, **kwargs):
        log_likelihoods = defaultdict(float)
        num_sentences = len(data)
        print_interval = int(num_sentences / 10000)
        self.init_translation_probabilities(mode=initialization, data=data, given_probs=kwargs.get("given_probs", None))

        for epoch in range(epochs):
            start = time.time()
            self.reset_counts()

            if verbosity > 0:
                print("\nStarting epoch #{}...".format(epoch+1))

            log_likelihoods[epoch] = self.expectation_step(
                data, epoch, print_interval=print_interval, verbosity=verbosity
            )
            self.maximization_step()

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

            if verbosity > 0 and self.eval_corpus is not None:
                aer = self.evaluate()
                print("AER for epoch #{} is {:.2f}.".format(epoch+1, aer))

        if verbosity > 0:
            sorted_translation_probs = sorted(self.translation_probs.items(), key=lambda tpl: tpl[1], reverse=True)
            for (source_token, target_token), prob in sorted_translation_probs[:50]:
                print("{} -> {}: {:.6f}".format(source_token, target_token, prob))

    @staticmethod
    def add_null_token(sentence):
        return ["NULL"] + sentence

    @abc.abstractmethod
    def reset_counts(self):
        pass

    @abc.abstractmethod
    def expectation_step(self, data, epoch, print_interval, verbosity):
        pass

    @abc.abstractmethod
    def maximization_step(self):
        pass

    def evaluate(self):
        if None in (self.eval_alignment_path, self.gold_standard, self.eval_corpus):
            raise AssertionError("Evaluation data is not given.")

        # Determine the models alignments via the viterbi algorithm
        predictions = []
        for (source_sentence, target_sentence) in self.eval_corpus:
            links = set()

            for target_pos, target_token in enumerate(target_sentence):
                translation_probs = [
                    self.translation_probs[(source_token, target_token)] for source_token in source_sentence
                ]
                source_pos = np.argmax(translation_probs)
                links.add((source_pos+1, target_pos+1))

            predictions.append(links)

        # Compute AER
        metric = AERSufficientStatistics()

        for gold_alignments, predictions in zip(self.gold_standard, predictions):
            metric.update(sure=gold_alignments[0], probable=gold_alignments[1], predicted=predictions)

        return metric.aer()


class Model1(Model):
    """
    Class for IBM model 1.
    """
    def __init__(self, epsilon, eval_alignment_path=None, eval_corpus=None):
        super().__init__(eval_alignment_path, eval_corpus)
        self.epsilon = epsilon  # Normalization constant

    def reset_counts(self):
        self.cooc_counts = defaultdict(float)
        self.source_counts = defaultdict(float)

    def expectation_step(self, data, epoch, print_interval=1, verbosity=0):
        log_likelihood = 0

        for i, (source_sentence, target_sentence) in enumerate(data):
            source_sentence = self.add_null_token(source_sentence)
            word_norms = defaultdict(float)
            sentence_log_likelihood = 0

            if verbosity > 0 and (i + 1) % print_interval == 0:
                print(
                    "\rProcessing sentence {}/{} ({:.2f} %)...".format(i + 1, len(data), (i + 1) / len(data) * 100),
                    end="", flush=True
                )

            # E-Step
            # Compute normalization
            # Implicitly uniform alignment probabilities as all alignments are considered equally
            for (source_token, target_token) in product(source_sentence, target_sentence):
                word_norms[source_token] += self.translation_probs[(source_token, target_token)]

            # Collect counts
            for source_token in source_sentence:
                source_token_probs = 0  # Sum of pi(f_j|e_i) over all i
                for target_token in target_sentence:
                    pair = (source_token, target_token)
                    delta = self.translation_probs[pair] / word_norms[source_token]
                    self.cooc_counts[pair] += delta
                    self.source_counts[source_token] += delta

                    # Accumulate (log-)likelihood for this sentence
                    source_token_probs += self.translation_probs[(source_token, target_token)]

                sentence_log_likelihood += np.log(source_token_probs)

            # Normalization of sentence log-likelihood by sentence lengths
            sentence_log_likelihood += np.log(self.epsilon) - len(target_sentence) * np.log(len(source_sentence))
            log_likelihood += sentence_log_likelihood

        return log_likelihood

    def maximization_step(self):
        # M-Step
        # Estimate probabilities
        for (source_type, target_type) in self.translation_probs.keys():
            pair = (source_type, target_type)
            try:
                self.translation_probs[pair] = self.cooc_counts[pair] / self.source_counts[source_type]
            except ZeroDivisionError:
                self.translation_probs[pair] = 0


class Model2(Model1):
    """
    Class for IBM model 2.
    """
    def __init__(self, epsilon, eval_alignment_path=None, eval_corpus=None):
        super().__init__(
            epsilon=epsilon, eval_alignment_path=eval_alignment_path, eval_corpus=eval_corpus
        )
        self.alignment_counts = defaultdict(float)  # c(j|i, m, l)
        self.aligned_counts = defaultdict(float)  # c(i, l, m)
        self.alignment_probs = AlignmentProbDict()

    def reset_counts(self):
        super().reset_counts()
        self.alignment_counts = defaultdict(float)
        self.aligned_counts = defaultdict(float)

    def expectation_step(self, data, epoch, print_interval=1, verbosity=0):
        log_likelihood = 0

        for i, (source_sentence, target_sentence) in enumerate(data):
            source_sentence = self.add_null_token(source_sentence)
            word_norms = defaultdict(float)
            sentence_log_likelihood = 0
            length_source = len(source_sentence)
            length_target = len(target_sentence)

            if verbosity > 0 and (i + 1) % print_interval == 0:
                print(
                    "\rProcessing sentence {}/{} ({:.2f} %)...".format(i + 1, len(data), (i + 1) / len(data) * 100),
                    end="", flush=True
                )

            # E-Step
            # Compute normalization
            # Implicitly uniform alignment probabilities as all alignments are considered equally
            pos_and_tokens = product(enumerate(source_sentence), enumerate(target_sentence))
            for (source_pos, source_token), (target_pos, target_token) in pos_and_tokens:
                word_norms[source_token] += self.translation_probs[
                    (source_token, target_token)
                ] * self.alignment_probs[(length_source, length_target, source_pos, target_pos)]

            # Collect counts
            for source_pos, source_token in enumerate(source_sentence):
                source_token_probs = 0  # Sum of pi(f_j|e_i) over all i
                for target_pos, target_token in enumerate(target_sentence):
                    pair = (source_token, target_token)
                    delta = self.translation_probs[pair] * self.alignment_probs[
                        (length_source, length_target, source_pos, target_pos)
                    ] / word_norms[source_token]
                    self.cooc_counts[pair] += delta
                    self.source_counts[source_token] += delta
                    source_token_probs += self.translation_probs[pair] * self.alignment_probs[
                        (length_source, length_target, source_pos, target_pos)
                    ]

                    self.alignment_counts[(length_source, length_target, source_pos, target_pos)] += delta
                    self.aligned_counts[(length_source, length_target, source_pos)] += delta

                sentence_log_likelihood += np.log(source_token_probs)

            # Normalization of sentence log-likelihood by sentence lengths
            sentence_log_likelihood += np.log(self.epsilon) - len(target_sentence) * np.log(len(source_sentence))
            log_likelihood += sentence_log_likelihood

        return log_likelihood

    def maximization_step(self):
        # Update translation probabilities
        super().maximization_step()

        # Update alignment probabilities
        for alignment_key in self.alignment_probs.keys():
            length_source, length_target, source_position, _ = alignment_key
            aligned_key = (length_source, length_target, source_position)
            self.alignment_probs[alignment_key] = self.alignment_counts[alignment_key] / self.aligned_counts[aligned_key]


if __name__ == "__main__":
    corpus = ParallelCorpus(
        source_path="./data/training/hansards.36.2.e", target_path="./data/training/hansards.36.2.f"
    )
    eval_corpus = ParallelCorpus(
        source_path="./data/validation/dev.e", target_path="./data/validation/dev.f"
    )

    #model1 = Model1(epsilon=0.1, eval_alignment_path="./data/validation/dev.wa.nonullalign", eval_corpus=eval_corpus)
    #model1.train(corpus, epochs=8)
    #model1.save("./model_iter20_eps01_uniform")
    #print(model1.translation_probs)

    model2 = Model2(epsilon=0.1, eval_alignment_path="./data/validation/dev.wa.nonullalign", eval_corpus=eval_corpus)
    model2.train(corpus, epochs=10)
    #model2.save("./model2_iter4_eps01_uniform")
