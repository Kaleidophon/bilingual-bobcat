
import codecs


class ParallelCorpus:
    def __init__(self, source_path, target_path):
        self.source_sentences = self.read_corpus_file(source_path)
        self.target_sentences = self.read_corpus_file(target_path)

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

