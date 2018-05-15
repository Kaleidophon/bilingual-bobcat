#### Proceprocessing
Preprocessing wasas done in the following order and follows he following naming scheme:

1. **Original file**: ``file.lang``
2. **Tokenizing**: ``file_tokenized.lang``
3. **Lower- and Truecasing**
    * **Lowercasing** ``file_lowercased.lang``
    * **Truecasing** ``file_truecased.lang``
        * To use truecasing, a truecasing model had to be trained first, for which a special corpus
        was used. This corpus is just the concatenation of the tokenized training, validation and test set.
        The name is ``truecasing_corpus.lang`` and the model is called `truecasing_model_lang`.
4. **Byte-Pair Encoding**
    * For the BPE, the encoding had to be trained as well, too. In the same manner as for truecasing, 
    a corpus was created by concatenating truecased training, validation and test set (``bpe_truecases_corpus.lang``).
    Resulting codes (``bpe_codes.lang``) were trained using at most 50k operations. The encoded files 
    are called ``file_bpe.lang``.