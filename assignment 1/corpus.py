"""
Module defining class and function relevant for data I/O and preprocessing.
"""

# STD
import codecs


class ParallelCorpus:
    def __init__(self, source_path, target_path):
        self.source_sentences = self.read_corpus_file(source_path)
        self.target_sentences = self.read_corpus_file(target_path)
        self.size = len(self.source_sentences)
        self.source_vocab = set()
        self._source_vocab_size = None

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