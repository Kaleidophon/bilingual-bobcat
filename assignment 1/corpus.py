"""
Module defining class and function relevant for data I/O and preprocessing.
"""

# STD
import codecs
from functools import reduce


class ParallelCorpus:
    """
    Class that contains a parallel corpus used for training IBM Model 1 and 2.
    """
    def __init__(self, source_path, target_path):
        # Enable a way to read multiple paths into one corpus if wanted
        source_paths = source_path if self.is_listlike(source_path) else (source_path, )
        target_paths = target_path if self.is_listlike(target_path) else (target_path, )

        self.source_sentences = self.read_all(source_paths)
        self.target_sentences = self.read_all(target_paths)
        self.size = len(self.source_sentences)
        self.source_vocab = set()
        self._source_vocab_size = None

    def read_all(self, paths):
        """
        Read all corpus file at given paths and merge results.
        """
        def _combine_lists(a, b):
            a.extend(b)
            return a

        sentences_per_path = [self.read_corpus_file(path) for path in paths]
        return reduce(_combine_lists, sentences_per_path)

    @staticmethod
    def read_corpus_file(path, filter_characters=[]):
        sentences = []

        with codecs.open(path, "rb", "utf-8") as corpus:
            for line in corpus.readlines():
                tokens = [token for token in line.strip().split() if token not in filter_characters]
                sentences.append(tokens)

        return sentences

    @property
    def parallel_sentences(self):
        return zip(self.source_sentences, self.target_sentences)

    @property
    def source_vocab_size(self):
        if self._source_vocab_size is None:
            self.source_vocab = {token for sentence in self.source_sentences for token in sentence}
            self._source_vocab_size = len(self.source_vocab)
        return self._source_vocab_size

    def __iter__(self):
        return (
            (source_sentence, target_sentence)
            for source_sentence, target_sentence in zip(self.source_sentences, self.target_sentences)
        )

    def __len__(self):
        return self.size

    @staticmethod
    def is_listlike(obj):
        return type(obj) in (tuple, list, set)
