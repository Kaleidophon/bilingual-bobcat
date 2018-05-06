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
import os

# EXT
import numpy as np
from scipy.special import digamma, gamma, loggamma
import matplotlib.pyplot as plt

# PROJECT
from aer import read_naacl_alignments, AERSufficientStatistics
from corpus import ParallelCorpus

# TODO
# - Jump distribution reparameterization


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
    def __init__(self, name=None, eval_alignment_path=None, eval_corpus=None, save_path=None):
        self.eval_alignment_path = eval_alignment_path
        self.gold_standard = None
        self.eval_corpus = eval_corpus
        self.name = name if name is not None else type(self).__name__
        self.save_path = save_path
        if save_path is not None:
            self.save_path = self.save_path if self.save_path.endswith("/") else self.save_path + "/"

        if self.eval_alignment_path is not None:
            self.gold_standard = read_naacl_alignments(self.eval_alignment_path)

        self.cooc_counts = defaultdict(float)  # p(e_l, f_k): Co-occurrences between source and target language words
        self.source_counts = defaultdict(float)  # Count of words in the source language
        self.translation_probs = defaultdict(lambda: defaultdict(float))

        # Save training metrics
        self.aers = []
        self.likelihoods = []

    def save(self, path):
        with open(path, "wb") as file:
            # You can't pickle lambda functions
            self.translation_probs = {key: dict(value) for key, value in self.translation_probs.items()}
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            model = pickle.load(file)
            return model

    def init_translation_probabilities(self, mode, data, given_probs):
        assert mode in ("uniform", "random", "continue"), "Invalid initialization mode {}".format(mode)
        translation_probs = defaultdict(lambda: defaultdict(float))

        # Initialize uniformly
        if mode == "uniform":
            translation_probs = defaultdict(lambda: defaultdict(lambda: 1 / data.source_vocab_size))

        # Initialize randomly
        elif mode == "random":
            translation_probs = defaultdict(lambda: defaultdict(lambda: random.random()))

        # Initialize with already trained probabilities
        elif mode == "continue":
            assert given_probs is not None
            translation_probs = given_probs

        return translation_probs

    def train(self, data: ParallelCorpus, epochs=10, initialization="uniform", verbosity=1, **kwargs):
        log_likelihoods = defaultdict(float)
        num_sentences = len(data)
        print_interval = int(num_sentences / 10000)
        self.translation_probs = self.init_translation_probabilities(
            mode=initialization, data=data, given_probs=kwargs.get("given_probs", None)
        )

        for epoch in range(epochs):
            start = time.time()
            self.reset_counts()

            if verbosity > 0:
                print("\nStarting epoch #{}...".format(epoch+1))

                log_likelihood = self.expectation_step(
                data, epoch, print_interval=print_interval, verbosity=verbosity
            )
            log_likelihoods[epoch] = log_likelihood
            self.likelihoods.append(log_likelihood)
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

            if self.eval_corpus is not None:
                aer = self.evaluate(verbosity=verbosity)
                self.aers.append(aer)

            # Save model every iteration if path is given
            if self.save_path is not None:
                self.save("{}{}_iter{}.pkl".format(self.save_path, self.name, epoch+1))

        self.plot_training()

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

    def evaluate(self, eval_alignment_path=None, eval_corpus=None, result_path=None, verbosity=1):
        # Overwrite evaluation data if necessary
        # (Helpful e.g. when you load a model and want to evaluate it on some given data)
        if eval_alignment_path is not None:
            self.eval_alignment_path = eval_alignment_path
            self.gold_standard = read_naacl_alignments(self.eval_alignment_path)
        if eval_corpus is not None:
            self.eval_corpus = eval_corpus

        # Delete result file if necessary
        if result_path is not None:
            if os.path.isfile(result_path):
                os.remove(result_path)

        if None in (self.eval_alignment_path, self.gold_standard, self.eval_corpus):
            raise AssertionError("Evaluation data is not given.")

        # Determine the models alignments via the viterbi algorithm
        predictions = []
        for sentence_no, (source_sentence, target_sentence) in enumerate(self.eval_corpus):
            links = set()

            for target_pos, target_token in enumerate(target_sentence):
                translation_probs = [
                    self.translation_probs[source_token][target_token] for source_token in source_sentence
                ]
                source_pos = np.argmax(translation_probs)
                links.add((source_pos+1, target_pos+1))

            predictions.append(links)

            # Write predictions into file if path to result file is given
            # Format: sentence_no position_L1 position_L2 [S P]
            if result_path is not None:
                with open(result_path, "a") as result_file:
                    for link in links:
                        result_file.write("{} {} {} S\n".format(str(sentence_no).zfill(4), *link))

        # Compute AER
        metric = AERSufficientStatistics()

        for gold_alignments, predictions in zip(self.gold_standard, predictions):
            metric.update(sure=gold_alignments[0], probable=gold_alignments[1], predicted=predictions)

        aer = metric.aer()

        if verbosity > 0:
            print("AER is {:.2f}.".format(aer))

        return aer

    @property
    def metrics(self):
        return {"AER": self.aers, "Log-likelihood": self.likelihoods}

    def plot_training(self):
        for metric_name, data in self.metrics.items():
            plt.figure()
            plt.plot(range(1, len(data)+1), data)
            plt.xlabel("Iteration")
            plt.xticks(range(1, len(data)+1))
            plt.ylabel(metric_name)
            plt.savefig("{}{}_{}.png".format(self.save_path, self.name.lower(), metric_name.lower()))


class Model1(Model):
    """
    Class for IBM model 1.
    """
    def __init__(self, epsilon, name=None, eval_alignment_path=None, eval_corpus=None, save_path=None):
        super().__init__(
            name=name, eval_alignment_path=eval_alignment_path, eval_corpus=eval_corpus, save_path=save_path
        )
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
                word_norms[source_token] += self.translation_probs[source_token][target_token]

            # Collect counts
            for source_token in source_sentence:
                source_token_probs = 0  # Sum of pi(f_j|e_i) over all i
                for target_token in target_sentence:
                    pair = (source_token, target_token)
                    delta = self.translation_probs[source_token][target_token] / word_norms[source_token]
                    self.cooc_counts[pair] += delta
                    self.source_counts[source_token] += delta

                    # Accumulate (log-)likelihood for this sentence
                    source_token_probs += self.translation_probs[source_token][target_token]

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
                self.translation_probs[pair] = 0  # TODO: This "fix" is wrong


class VariationalModel1(Model1):
    """
    IBM model 1 using a variational Bayes approach.
    """
    def __init__(self, alpha, epsilon, name=None, eval_alignment_path=None, eval_corpus=None, save_path=None):
        super().__init__(
            epsilon=epsilon, name=name, eval_alignment_path=eval_alignment_path, eval_corpus=eval_corpus,
            save_path=save_path
        )
        self.alpha = alpha  # Dirichlet prior belief
        self.lambdas = defaultdict(lambda: defaultdict(lambda: self.alpha))  # lambda_f|e

        self.elbos = []  # Record ELBO per epoch

    def reset_counts(self):
        pass

    def expectation_step(self, data, epoch, print_interval=1, verbosity=0):
        log_likelihood = 0

        # E-Step
        # Update estimated translation probabilities
        number_types = len(self.translation_probs)
        for i, source_type in enumerate(self.translation_probs.keys()):
            if verbosity > 0:
                print(
                    "\rUpating translation probs for type {}/{} ({:.2f} %)...".format(
                        i + 1, number_types, (i + 1) / number_types * 100
                    ), end="", flush=True
                )

            source_norm = digamma(sum(self.lambdas[source_type].values()))
            for target_type in self.translation_probs[source_type].keys():
                try:
                    self.translation_probs[source_type][target_type] = np.exp(
                        digamma(self.lambdas[source_type][target_type]) - source_norm
                    )
                except TypeError:
                    print(source_type, target_type)
                    print(self.lambdas[(source_type, target_type)])
                    print(sum(self.lambdas[source_type].values()))

        # Reset lambdas here
        self.lambdas = defaultdict(lambda: defaultdict(lambda: self.alpha))  # lambda_f|e

        # Accumulate counts
        for i, (source_sentence, target_sentence) in enumerate(data):
            source_sentence = self.add_null_token(source_sentence)
            word_norms = defaultdict(float)
            sentence_log_likelihood = 0

            if verbosity > 0 and (i + 1) % print_interval == 0:
                print(
                    "\rProcessing sentence {}/{} ({:.2f} %)...".format(i + 1, len(data), (i + 1) / len(data) * 100),
                    end="", flush=True
                )

            # Compute normalization
            # Implicitly uniform alignment probabilities as all alignments are considered equally
            for (source_token, target_token) in product(source_sentence, target_sentence):
                word_norms[source_token] += self.translation_probs[source_token][target_token]

            # Collect counts
            for source_token in source_sentence:
                source_token_probs = 0  # Sum of pi(f_j|e_i) over all i
                for target_token in target_sentence:
                    delta = self.translation_probs[source_token][target_token] / word_norms[source_token]

                    # Cheat a little here and already update lambdas
                    self.lambdas[source_token][target_token] += delta

                    # Accumulate (log-)likelihood for this sentence
                    source_token_probs += self.translation_probs[source_token][target_token]

                sentence_log_likelihood += np.log(source_token_probs)

            # Normalization of sentence log-likelihood by sentence lengths
            sentence_log_likelihood += np.log(self.epsilon) - len(target_sentence) * np.log(len(source_sentence))
            log_likelihood += sentence_log_likelihood

        elbo = self.elbo(log_likelihood)
        self.elbos.append(elbo)

        if verbosity > 0:
            print("\nELBO for epoch #{} is {:.4f}".format(epoch+1, elbo))

        return log_likelihood

    def maximization_step(self):
        # M-Step
        # This is for convenience already done in the E-step, this function is only here for consistency
        pass

    def elbo(self, log_likelihood):
        """
        Calculate the expected lower bound for the variational bayes model.
        """
        divergence_sum = 0

        for source_type in self.translation_probs.keys():
            source_norm = digamma(sum(self.lambdas[source_type].values()))
            target_types = self.translation_probs[source_type].keys()

            theta_sum = 0  # Sum over all probabilities theta_f|e and other terms for all f in V_f
            alpha_sum = 0  # Sum over all alpha_f for all f in V_f
            lambda_sum = 0  # Sum over all lambda_f|e for all f in V_f

            for target_type in target_types:
                current_lambda = self.lambdas[source_type][target_type]
                sufficient_statistic = digamma(current_lambda) - source_norm

                # Update sums (using constant alpha_f for all f in V_f)
                theta_sum += sufficient_statistic * (self.alpha - current_lambda) + loggamma(current_lambda) - loggamma(self.alpha)
                alpha_sum += self.alpha
                lambda_sum += current_lambda

            # We calculate the KL divergence for every source type and sum them together in the end
            kl_divergence = theta_sum + loggamma(alpha_sum) - loggamma(lambda_sum)
            divergence_sum += kl_divergence

        return log_likelihood - divergence_sum.real

    @property
    def metrics(self):
        return {"AER": self.aers, "Log-likelihood": self.likelihoods, "ELBO": self.elbos}

    def save(self, path):
        with open(path, "wb") as file:
            # You can't pickle lambda functions
            self.translation_probs = {key: dict(value) for key, value in self.translation_probs.items()}
            self.lambdas = {key: dict(value) for key, value in self.lambdas.items()}
            pickle.dump(self, file)


class Model2(Model1):
    """
    Class for IBM model 2.
    """
    def __init__(self, epsilon, name=None, eval_alignment_path=None, eval_corpus=None, save_path=None):
        super().__init__(
            epsilon=epsilon, name=name, eval_alignment_path=eval_alignment_path, eval_corpus=eval_corpus,
            save_path=save_path
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
                word_norms[source_token] += self.translation_probs[source_token][target_token] * self.alignment_probs[
                    (length_source, length_target, source_pos, target_pos)
                ]

            # Collect counts
            for source_pos, source_token in enumerate(source_sentence):
                source_token_probs = 0  # Sum of pi(f_j|e_i) over all i
                for target_pos, target_token in enumerate(target_sentence):
                    pair = (source_token, target_token)
                    delta = self.translation_probs[source_token][target_token] * self.alignment_probs[
                        (length_source, length_target, source_pos, target_pos)
                    ] / word_norms[source_token]
                    self.cooc_counts[pair] += delta
                    self.source_counts[source_token] += delta
                    source_token_probs += self.translation_probs[source_token][target_token] * self.alignment_probs[
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
        source_path=["./data/training/hansards.36.2.e", "./data/validation/dev.e"],
        target_path=["./data/training/hansards.36.2.f", "./data/validation/dev.f"]
    )
    eval_corpus = ParallelCorpus(
        source_path="./data/validation/dev.e", target_path="./data/validation/dev.f"
    )
    test_corpus = ParallelCorpus(
        source_path="./data/testing/test/test.e", target_path="./data/testing/test/test.f"
    )
    test_alignments = "./data/testing/answers/test.wa.nonullalign"

    #model1 = Model1(epsilon=0.1, eval_alignment_path="./data/validation/dev.wa.nonullalign", eval_corpus=eval_corpus)
    #model1.train(corpus, epochs=10, initialization="random")
    #model1.save("./model_iter10_eps01_uniform")
    #print(model1.translation_probs)
    # model1 = Model1.load("./model_iter10_eps01_uniform")
    #
    model2 = Model2(
        name="model2", save_path="./models/",
        epsilon=0.1, eval_alignment_path="./data/validation/dev.wa.nonullalign", eval_corpus=eval_corpus
    )
    model2.train(corpus, epochs=3, initialization="uniform")
    #model2.train(corpus, epochs=10, initialization="continue", given_probs=model1.translation_probs)

    # varmodel1 = VariationalModel1(
    #     name="varbayes", save_path="./models/",
    #     alpha=0.1, epsilon=0.1, eval_alignment_path="./data/validation/dev.wa.nonullalign", eval_corpus=eval_corpus
    # )
    # varmodel1.train(corpus, epochs=3, initialization="random")

    # # Evaluate
    # print("Evaluating model 1...")
    # model1 = Model1.load("./model_iter10_eps01_uniform")
    # model1.evaluate(
    #     eval_alignment_path=test_alignments, eval_corpus=test_corpus, result_path="./eval_out/ibm1.mle.naacl"
    # )
    #
    # print("Evaluating model 2...")
    # model2 = Model2.load("./model2_iter10_eps01_continue")
    # model2.evaluate(
    #     eval_alignment_path=test_alignments, eval_corpus=test_corpus, result_path="./eval_out/ibm2.mle.naacl"
    # )
