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
    def __init__(self, source_path, target_path, max_sentence_length=50, max_source_vocab_size=np.inf,
                 max_target_vocab_size=np.inf, use_indices_from=None):
        if use_indices_from:
            assert type(use_indices_from) == type(self), "You can only use indices from another ParallelCorpus class"

        # Enable a way to read multiple paths into one corpus if wanted
        # Read in all the data
        self.max_sentence_length = max_sentence_length
        self.source_sentences, skip_sentences = self.read_corpus_file(source_path, self.max_sentence_length)
        self.target_sentences, _ = self.read_corpus_file(
            target_path, self.max_sentence_length, other_skip_sentences=skip_sentences
        )
        self.size = len(self.source_sentences)

        # Index corpus and create vocabulary
        indexed_source = self.index_corpus(
            self.source_sentences, max_vocab_size=max_source_vocab_size,
            given_word2idx=None if not use_indices_from else use_indices_from.source_w2i
        )
        self.source_idx, self.source_vocab, self.source_w2i, self.source_i2w, self.source_vocab_size, self.source_lengths = indexed_source
        indexed_target = self.index_corpus(
            self.target_sentences, add_sentence_delimiters=True, max_vocab_size=max_target_vocab_size,
            given_word2idx=None if not use_indices_from else use_indices_from.target_w2i
        )
        self.target_idx, self.target_vocab, self.target_w2i, self.target_i2w, self.target_vocab_size, self.target_lengths = indexed_target

        # Create position vectors
        self.pos_base = [i for i in range(1, max_sentence_length + 1)]
        self.positions = [self.pos_base[:i] for i in self.source_lengths]
        self.source_pad = self.source_w2i['<pad>']
        self.target_pad = self.target_w2i['<pad>']

        # Pad the sentences and positions - useful in order to convert everything to tensors and train batch
        self.source_padded = self.pad_sequences(self.source_idx, self.source_lengths, self.source_w2i['<pad>'])
        self.target_padded = self.pad_sequences(self.target_idx, self.target_lengths, self.target_w2i['<pad>'])
        self.positions = self.pad_sequences(self.positions, self.source_lengths, 0)
        self.source_sentence_ids = np.array(range(len(self.source_sentences)))
        self.target_sentence_ids = np.array(range(len(self.source_sentences)))

        # Convert everything into tensors for pytorch and sort by sentence size, descending
        data = [
            self.source_padded, self.target_padded, self.source_lengths, self.target_lengths, self.positions,
            self.source_sentence_ids, self.target_sentence_ids
        ]
        tensors = list(map(self.convert_to_tensor, data))
        sorted_tensors = self.sort_tensors(*tensors)
        self.source_tensor, self.target_tensor, self.source_lengths, self.target_lengths, self.positions, \
            self.source_sentence_ids, self.target_sentence_ids = sorted_tensors

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
    def read_corpus_file(path, max_sentence_length, other_skip_sentences=set(), filter_characters=[]):
        sentences = []
        skip_sentences = set()  # Keep track of sentences which are skipped due to length so they can be skipped in the
        # other corpus file too

        with codecs.open(path, "rb", "utf-8") as corpus:
            for i, line in enumerate(corpus.readlines()):
                tokens = [token for token in line.strip().split() if token not in filter_characters]

                # Filter out sentences that are too long if you're reading the first part of the parallel corpus
                # Or filter them out if they have been filtered out in the previous part of the corpus
                if (len(tokens) <= max_sentence_length and len(other_skip_sentences) == 0) \
                        or i not in other_skip_sentences:
                    sentences.append(tokens)
                else:
                    skip_sentences.add(i)

        return sentences, skip_sentences

    def index_corpus(self, sentences, max_vocab_size=np.inf, add_sentence_delimiters=False, given_word2idx=None):
        # Use indices create on other set if given
        word2idx = self.init_word2idx() if not given_word2idx else given_word2idx
        indexed_sentences = []
        sentence_lengths = []
        token_freqs = defaultdict(int)

        for line in sentences:
            current_indexed_sentence = [] if not add_sentence_delimiters else [word2idx["<bos>"]]

            for word in line:
                token_freqs[word] += 1
                current_indexed_sentence.append(word2idx[word])

            if add_sentence_delimiters:
                current_indexed_sentence.append(word2idx["<eos>"])

            indexed_sentences.append(current_indexed_sentence)

            sentence_length = len(line) if not add_sentence_delimiters else len(line) + 2
            sentence_lengths.append(sentence_length)

        # Remove infrequent words (they will be mapped to <unk> later) until max_vocab size is reached
        vocab_size = len(word2idx)
        sorted_token_freqs = sorted(list(token_freqs.items()), key=lambda x: x[1])

        while len(sorted_token_freqs) > max_vocab_size:
            infrequent_word, _ = sorted_token_freqs.pop(0)
            del word2idx[infrequent_word]

        vocab = set(word2idx.keys())

        idx2word = defaultdict(lambda: "<unk>", {idx: w for (w, idx) in word2idx.items()})

        # After reading the data, unknown word just return the index of the <unk> token (don't generate new indices)
        word2idx = defaultdict(lambda: word2idx["<unk>"], word2idx)

        return indexed_sentences, vocab, word2idx, idx2word, vocab_size, sentence_lengths

    @staticmethod
    def init_word2idx():
        word2idx = defaultdict(lambda: len(word2idx))
        _ = word2idx["<pad>"], word2idx["<unk>"], word2idx["<bos>"], word2idx["<eos>"]
        return word2idx

    @staticmethod
    def pad_sequences(sentence_idx, seq_lengths, pad):
        # Fill everything with padding first
        padding = np.full((len(sentence_idx), max(seq_lengths)), pad)

        # Replace with actual token ids wherever possible
        for idx, (seq, seqlen) in enumerate(zip(sentence_idx, seq_lengths)):
            padding[idx, :seqlen] = seq

        return padding

    @staticmethod
    def convert_to_tensor(data):
        if torch.cuda.is_available():
            return torch.LongTensor(data).cuda()
        return torch.LongTensor(data)

    @staticmethod
    def sort_tensors(source, target, source_lengths, target_lengths, pos, source_ids, target_ids):
        target_lengths, perm_idx = target_lengths.sort(0, descending=True)
        source_tensor = source[perm_idx]
        target_tensor = target[perm_idx]
        source_lengths = source_lengths[perm_idx]
        positions = pos[perm_idx]
        source_ids = source_ids[perm_idx]
        target_ids = source_ids[perm_idx]
        return source_tensor, target_tensor, source_lengths, target_lengths, positions, source_ids, target_ids

    def __len__(self):
        return self.source_tensor.size(0)

    def __getitem__(self, index):
        return self.source_tensor[index], self.target_tensor[index], self.source_lengths[index], \
               self.target_lengths[index], self.positions[index]

    @staticmethod
    def is_listlike(obj):
        return type(obj) in (tuple, list, set)