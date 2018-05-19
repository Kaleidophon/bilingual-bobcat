"""
Module defining class and function relevant for data I/O and preprocessing.
"""

# STD
import codecs
from functools import reduce
from collections import defaultdict

# EXT
import numpy as np
import torch
from torch.utils.data import Dataset


class ParallelCorpus(Dataset):
    """
    Class that contains a parallel corpus used for training IBM Model 1 and 2.
    """
    def __init__(self, source_path, target_path, max_sentence_length=50):
        # Enable a way to read multiple paths into one corpus if wanted
        source_paths = source_path if self.is_listlike(source_path) else (source_path, )
        target_paths = target_path if self.is_listlike(target_path) else (target_path, )

        # Read in all the data
        self.max_sentence_length = max_sentence_length
        self.source_sentences = self.read_all(source_paths)
        self.target_sentences = self.read_all(target_paths)
        self.size = len(self.source_sentences)

        # Index corpus and create vocabulary
        indexed_source = self.index_corpus(self.source_sentences)
        self.source_idx, self.source_vocab, self.source_w2i, self.source_i2w, self.source_vocab_size, self.source_lengths = indexed_source
        indexed_target = self.index_corpus(self.target_sentences)
        self.target_idx, self.target_vocab, self.target_w2i, self.target_i2w, self.target_vocab_size, self.target_lengths = indexed_target

        # Create position vectors
        self.pos_base = [i for i in range(1, max_sentence_length + 1)]
        self.positions = [self.pos_base[:i] for i in self.source_lengths]

        # Pad the sentences and positions - useful in order to convert everything to tensors and train batch
        self.source_padded = self.pad_sequences(self.source_idx, self.source_lengths, self.source_w2i['<pad>'])
        self.target_padded = self.pad_sequences(self.target_idx, self.target_lengths, self.target_w2i['<pad>'])
        self.positions = self.pad_sequences(self.positions, self.source_lengths, 0)

        # Convert everything into tensors for pytorch and sort by sentence size, descending
        data = [self.source_padded, self.target_padded, self.source_lengths, self.target_lengths, self.positions]
        tensors = list(map(self.convert_to_tensor, data))
        sorted_tensors = self.sort_tensors(*tensors)
        self.source_tensor, self.target_tensor, self.source_lengths, self.target_lengths, self.positions = sorted_tensors

    def read_all(self, paths):
        """
        Read all corpus file at given paths and merge results.
        """
        def _combine_lists(a, b):
            a.extend(b)
            return a

        sentences_per_path = [self.read_corpus_file(path, self.max_sentence_length) for path in paths]
        return reduce(_combine_lists, sentences_per_path)

    @staticmethod
    def read_corpus_file(path, max_sentence_length, filter_characters=[]):
        sentences = []

        with codecs.open(path, "rb", "utf-8") as corpus:
            for line in corpus.readlines():
                tokens = [token for token in line.strip().split() if token not in filter_characters]

                # Filter out sentences that are too long
                if len(tokens) <= max_sentence_length:
                    sentences.append(tokens)

        return sentences

    def index_corpus(self, sentences):
        word2idx = self.init_word2idx()
        indexed_sentences = []
        sentence_lengths = []

        for line in sentences:
            current_indexed_sentence = []

            for word in line:
                current_indexed_sentence.append(word2idx[word])

            indexed_sentences.append(current_indexed_sentence)
            sentence_lengths.append(len(line))

        idx2word = {idx: w for (w, idx) in enumerate(word2idx)}

        vocab_size = len(word2idx)
        vocab = set(word2idx.keys())

        return indexed_sentences, vocab, word2idx, idx2word, vocab_size, sentence_lengths

    @staticmethod
    def init_word2idx():
        word2idx = defaultdict(lambda: len(word2idx))
        _ = word2idx["<pad>"], word2idx["<unk>"]
        return word2idx

    @staticmethod
    def pad_sequences(sentence_idx, seq_lengths, pad):
        # Fill everything with padding first
        padding = np.full((len(sentence_idx), max(seq_lengths)), pad)

        # Replace with actual token ids whereever possible
        for idx, (seq, seqlen) in enumerate(zip(sentence_idx, seq_lengths)):
            padding[idx, :seqlen] = seq

        return padding

    @staticmethod
    def convert_to_tensor(data):
        if torch.cuda.is_available():
            return torch.LongTensor(data).cuda()
        return torch.LongTensor(data)

    @staticmethod
    def sort_tensors(source, target, source_lengths, target_lengths, pos):
        target_lengths, perm_idx = target_lengths.sort(0, descending=True)
        source_tensor = source[perm_idx]
        target_tensor = target[perm_idx]
        source_lengths = source_lengths[perm_idx]
        positions = pos[perm_idx]
        return source_tensor, target_tensor, source_lengths, target_lengths, positions

    def __len__(self):
        return self.source_tensor.size(0)

    def __getitem__(self, index):
        return self.source_tensor[index], self.target_tensor[index], self.source_lengths[index], \
               self.target_lengths[index]

    @staticmethod
    def is_listlike(obj):
        return type(obj) in (tuple, list, set)


if __name__ == "__main__":
    corpus = ParallelCorpus(source_path="./data/train/train_bpe.en", target_path="./data/train/train_bpe.fr")