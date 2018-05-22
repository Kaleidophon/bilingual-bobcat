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
from models import AttentionModel


def train(model, num_epochs, loss_function, optimizer, target_dim, save_dir=None):
    epoch_losses = []
    iterations = len(data_loader)

    for epoch in range(0, num_epochs):
        start = time.time()
        batch_losses = []
        batch = 0

        for source_batch, target_batch, source_lengths, target_lengths, batch_positions in data_loader:
            batch_start = time.time()
            max_len = int(source_lengths.max().cpu().numpy())
            source_batch = source_batch[:, :max_len]
            batch_positions = batch_positions[:, :max_len]
            target_len = target_lengths.max()

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

            combined_embeddings, hidden = model.encoder_forward(source_batch, batch_positions)
            loss = 0

            # remove then starting token for evaluation, 
            pad_tensor = torch.zeros(target_batch.size(0),1).long().cuda()
            target_batch2 = target_batch[:,1:]
            target_batch = torch.cat((target_batch,pad_tensor),1)

            for target_pos in range(target_len):
                current_target_words = target_batch[:, target_pos]
                #print("Current target words", current_target_words)
                out, hidden = model.decoder_forward(
                    current_target_words, hidden, combined_embeddings, source_lengths, max_len
                )

                # remove the start token from the targets and the end token from the decoder
                #out = out[:, :-1]
                #target_batch = target_batch[:, 1:out.size(1)]
                #print("outs2")
                #print(out.size(), target_batch.size())

                # pytorch expects the inputs for the loss function to be: (batch x classes)
                # cant handle sentences so we just transform our data to treath each individual word as a batch:
                # so new batch size = old batch * padded sentence length
                # should not matter as it all gets averaged out anyway
                #batch_size = out.size(0)
                #out = out.view(batch_size * max_len, out.size(1))
                #print("Reshaped out", out.size())
                #target_batch = target_batch.view(batch_size * max_len, target_batch.size(1))

                #print("Loss", out.size(), target_onehots.size())
                #print("Loss in", out.size(), target_batch[:, target_pos].size())

                # TODO: Remove BOS and EOS tokens from loss calculation
                # Loss w/o <bos> and <eos> token
                print("loss", out.size(), target_batch.size())
                #if 0 < target_pos < out.size(1)-1:
                # TODO: Some RuntimeError because in batch 78
                try:
                    loss += loss_function(out, target_batch2[:, target_pos])
                except RuntimeError:
                    print("\nloss", target_pos, out[:, :-1].size(), target_batch.size(), target_batch[:, target_pos].size(), "\n")

            batch_losses.append(loss)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time = time.time() - batch_start
            #print('\r[Epoch {:03d}/{:03d}] Batch {:06d}/{:06d} [{:.1f}/s] '.format(epoch + 1, num_epochs, batch + 1,
            #                                                                       iterations, batch_time), end='')
            #batch += 1

        avg_loss = sum(batch_losses) / iterations
        epoch_losses.append(avg_loss)
        #print('Time: {:.1f}s Loss: {:.3f} score2?: {:.6f}'.format(time.time() - start, avg_loss, 0))

        if save_dir is not None:
            model.save("{}{}_epoch{}.model".format(save_dir, model.__class__.__name__.lower(), epoch+1))


def convert_to_one_hot(word_indices, target_dim):
    y = word_indices % target_dim
    batch_size, _ = word_indices.size()
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot = torch.LongTensor(batch_size, target_dim)

    if torch.cuda.is_available():
    	y_onehot = y_onehot.cuda()

    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot


if __name__ == "__main__":
    # Define hyperparameters
    num_epochs = 30
    batch_size = 128
    learning_rate = 0.01
    embedding_dim = 100
    hidden_dim = 2 * embedding_dim
    max_allowed_sentence_len = 50

    # Prepare training
    training_set = ParallelCorpus(
        source_path="./data/train/train_bpe.fr", target_path="./data/train/train_bpe.en",
        max_sentence_length=max_allowed_sentence_len
    )
    data_loader = DataLoader(training_set, batch_size=batch_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=training_set.target_pad).cpu()
    max_corpus_sentence_len = training_set.source_lengths.max().cpu().numpy()

    # Init model
    model = AttentionModel(
        source_vocab_size=training_set.source_vocab_size, target_vocab_size=training_set.target_vocab_size,
        embedding_dim=embedding_dim, hidden_dim=hidden_dim, encoded_positions=51
    )
    optimizer = torch.optim.Adam(list(model.parameters()), learning_rate)

    # Train
    train(model, num_epochs, loss_function, optimizer, training_set.target_vocab_size, save_dir="./")

