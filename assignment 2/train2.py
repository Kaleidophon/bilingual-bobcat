"""
Module defining the model training function.
"""

# STD
import time

# EXT
import torch
from torch import nn
from torch.utils.data import DataLoader

# PROJECT
from corpus import ParallelCorpus
from models2 import Decoder, Encoder


def train(encoder, decoder, num_epochs, loss_function, optimizer, target_dim, force, iterations, save_dir=None):
    epoch_losses = []

    for epoch in range(0, num_epochs):
        start = time.time()
        batch_loss = 0
        batch = 0

        for source_batch, target_batch, source_lengths, target_lengths, batch_positions in data_loader:
            batch_start = time.time()
            max_len = source_lengths.max()
            source_batch = source_batch[:, :max_len]
            batch_positions = batch_positions[:, :max_len]
            target_len = target_lengths.max()
            target_batch = target_batch[:,:target_len]

            source_batch = torch.autograd.Variable(source_batch)
            target_batch = torch.autograd.Variable(target_batch)
            source_lengths = torch.autograd.Variable(source_lengths)
            target_lengths = torch.autograd.Variable(target_lengths)
            batch_positions = torch.autograd.Variable(batch_positions)

            if torch.cuda.is_available():
                source_batch = source_batch.cuda()
                target_batch = target_batch.cuda()
                source_lengths = source_lengths.cuda()
                target_lengths = target_lengths.cuda()
                batch_positions = batch_positions.cuda()

            loss = 0

            # get encoder outputs
            encoder_out, hidden = encoder(source_batch, batch_positions)

            use_teacher_forcing = True if random.random() <= force else False

            # BOTH LIKELY TO GO OUT OF BOUNDS DUE TO THE +1 IN THE LOSS CALC LINE.
            # EITHER APPEND A PAD LINE TO THE TARGET BATCH AT THE END HERE OR
            # OR ADJUST CORPUS.PY TO ADD AN EXTRA PAD TO EACH SENTENCE.
            if use_teacher_forcing:
                for i in range(target_len):
                    word_batch = target_batch[:,i,:]
                    decoder_out, hidden = decoder(word_batch, hidden, target_lengths, encoder_out, source_lengths, max_len)
                    loss += loss_function(decoder_out, target_batch[:,i+1,:])

            else:
                decoder_in = target_batch[:,0,:]
                for i in range(target_len):
                    # TODO: METHOD TO LIMIT TO SENTENCES STILL ACTIVE IN i (ie NOT PADDING RIGHT NOW)
                    # PERHAPS NOT NEEDED AS LOSS INGNORES PADS ANYWAY???
                    decoder_out, hidden = decoder(decoder_in, hidden, target_lengths, encoder_out, source_lengths, max_len)
                    loss += loss_function(decoder_out, target_batch[:,i,:])
                    _, words = decoder_out.topk(1)
                    decoder_in = words.squeeze().detach() # this is not tested.

            batch_loss += loss / target_lengths

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time = time.time() - batch_start
            print('\r[Epoch {:03d}/{:03d}] Batch {:06d}/{:06d} [{:.1f}/s] '.format(epoch + 1, num_epochs, batch + 1,
                                                                                   iterations, batch_time), end='')
            batch += 1

        epoch_losses.append(avg_loss)
        print('Time: {:.1f}s Loss: {:.3f}'.format(time.time() - start, batch_loss))

        if save_dir is not None:
            torch.save(model, "{}{}_epoch{}.model".format(save_dir, model.__class__.__name__.lower(), epoch+1))


if __name__ == "__main__":
    # Define hyperparameters
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.01
    embedding_dim = 100
    hidden_dim = 2 * embedding_dim
    max_allowed_sentence_len = 50
    force = 1

    # Prepare training
    training_set = ParallelCorpus(
        source_path="./data/train/train_bpe.fr", target_path="./data/train/train_bpe.en",
        max_sentence_length=max_allowed_sentence_len
    )
    data_loader = DataLoader(training_set, batch_size=batch_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=training_set.target_pad)
    iterations = len(data_loader)
    # Init model
    encoder = Decoder(
        source_vocab_size=training_set.source_vocab_size, embedding_dims=embedding_dim,
        max_sentence_len=max_allowed_sentence_len, hidden_dims=hidden_dim
    )

    decoder = Decoder(
        target_vocab_size=training_set.target_vocab_size,
        embedding_dims=embedding_dim, hidden_dims=hidden_dim
    )
    if torch.cuda.is_available():
        loss_function = loss_function.cuda()
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    optimizer = torch.optim.Adam(list(encoder.parameters(),decoder.parameters()), learning_rate)


    # Train
    train(encoder, decoder, num_epochs, loss_function, optimizer, training_set.target_vocab_size, force, iterations, save_dir="./")

