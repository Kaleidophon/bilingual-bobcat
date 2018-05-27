"""
This is just defined here so we can load it inside the eval.py module. Replace the code with the most recent versions!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class Seq2Seq(nn.Module):
    """A Vanilla Sequence to Sequence (Seq2Seq) model with LSTMs.
    Ref: Sequence to Sequence Learning with Neural Nets
    https://arxiv.org/abs/1409.3215
    """

    def __init__(
        self, trg_emb_dim,
        trg_vocab_size, trg_hidden_dim,
        pad_token_trg, drop, context=True,
        LSTM_instead=False, bidirectional=False,
        nlayers_trg=1,
    ):
        """Initialize Seq2Seq Model."""
        super(Seq2Seq, self).__init__()
        self.trg_vocab_size = trg_vocab_size
        self.trg_emb_dim = trg_emb_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.bidirectional = bidirectional
        self.nlayers_trg = nlayers_trg
        self.pad_token_trg = pad_token_trg
        self.attn_soft = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.LSTM_instead=LSTM_instead
        self.context = context

        # Word Embedding look-up table for the target language
        self.trg_embedding = nn.Embedding(
            self.trg_vocab_size,
            self.trg_emb_dim,
            self.pad_token_trg,
        )


        # Decoder GRU // LSTM
        if (not self.LSTM_instead):
        
            self.decoder = nn.GRU(
                self.trg_emb_dim,
                self.trg_hidden_dim,
                self.nlayers_trg,
                batch_first=True
            )
        else:
            self.decoder = nn.LSTM(
                self.trg_emb_dim,
                self.trg_hidden_dim,
                self.nlayers_trg,
                batch_first=True
            )
        
#         self.scaler = nn.Linear(
#             self.trg_hidden_dim,
#             self.trg_emb_dim*2,
#         )
        
#         self.scaler2 = nn.Linear(
#             self.trg_hidden_dim,
#             self.trg_emb_dim,
#         )
        
        # Projection layer from decoder hidden states to target language vocabulary
        
        if (not self.context):
            self.decoder2vocab = nn.Linear(trg_hidden_dim, trg_vocab_size)
        else:
            self.decoder2vocab = nn.Linear(trg_hidden_dim + trg_emb_dim*2, trg_vocab_size)

    def forward(self, encoder_out, h_t, input_trg, source_lengths, teacher ):
        trg_emb = self.trg_embedding(input_trg)
        trg_emb = self.dropout(trg_emb)
        
        h_t = h_t.unsqueeze(0).expand(self.nlayers_trg, h_t.size(0), h_t.size(1))
        if (self.LSTM_instead):
            h_t = (h_t,h_t)
        
        hidden = []
        outputs = []
        trg_in = trg_emb[:,0,:].unsqueeze(1)
        for i in range(input_trg.size(1)):
#             print( " ")
            if (teacher):
                trg_in = trg_emb[:,i,:].unsqueeze(1)
#             print(trg_in.size())
            trg_h, h_t = self.decoder(trg_in, h_t)
            hidden.append(h_t.squeeze())
#             print(h_t.size())

            if (self.context):
                context = self.attention(h_t.squeeze().unsqueeze(1),encoder_out,source_lengths, encoder_out.size(1))
                trg_h = torch.cat((trg_h,context),2)
            
            trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1), trg_h.size(2)
            )
            # Affine transformation of all decoder hidden states
            decoder2vocab = self.decoder2vocab(trg_h_reshape)
            # Reshape
            decoder2vocab = decoder2vocab.view(
                trg_h.size(0), trg_h.size(1), decoder2vocab.size(1)
            )

        
            
            outputs.append(decoder2vocab.squeeze())
            
            if (not teacher):
                _, word = torch.max(decoder2vocab,2)
                trg_in = self.trg_embedding(word)
            
        outputs = torch.stack(outputs,1)
        return outputs    
            
            
            
#             if (not teacher):
#                 trg_in = self.scaler2(decoder_out)
#                 trg_in = self.relu(trg_in)
            
        # hiddens = torch.stack(hidden,1)
        # trg_h = torch.stack(outputs,1)
#         print(trg_h.size())
#         print(hiddens.size())
        #if torch.cuda.is_available()
        #    trg_
        #print(asdfs)
        
            
        #hiddens = self.scaler(hiddens) 
        #hiddens = self.relu(hiddens)
#         if (self.context):
#             context = self.attention(hiddens,encoder_out,source_lengths, encoder_out.size(1))
#             print(hiddens.size())
#             print(encoder_out.size())

#             trg_h = torch.cat((trg_h,context),2)
        
        
#         # Initialize the decoder GRU with the last hidden state of the encoder and 
#         # run target inputs through the decoder.
        
#         # Merge batch and time dimensions to pass to a linear layer
#         trg_h_reshape = trg_h.contiguous().view(
#             trg_h.size(0) * trg_h.size(1), trg_h.size(2)
#         )
        
#         # Affine transformation of all decoder hidden states
#         decoder2vocab = self.decoder2vocab(trg_h_reshape)
        
#         # Reshape
#         decoder2vocab = decoder2vocab.view(
#             trg_h.size(0), trg_h.size(1), decoder2vocab.size(1)
#         )
    
    
    def attention(self, hidden_to_attn, encoder_outputs, source_lengths, max_len):
                
        
#         print(hidden_to_attn.size())
#         print(encoder_outputs.size())
        # repeat the lstm out in third dimension and the encoder outputs in second dimension so we can make a meshgrid
        # so we can do elementwise mul for all possible combinations of h_j and s_i
        h_j = encoder_outputs.unsqueeze(1).repeat(1,hidden_to_attn.size(1),1,1)
        s_i = hidden_to_attn.unsqueeze(2).repeat(1,1,encoder_outputs.size(1),1)
#         print(h_j.size())
#         print(s_i.size())
        # get the dot product between the two to get the energy
        # the unsqueezes are there to emulate transposing. so we can use matmul as torch.dot doesnt accept matrices
        energy = s_i.unsqueeze(3).matmul(h_j.unsqueeze(4)).squeeze(4)
        
#         # this is concat attention, its a different form then the ones we need
#         cat = torch.cat((s_i,h_j),3)
        
#         energy = self.attn_layer(cat)

        # reshaping the encoder outputs for later
        encoder_outputs = encoder_outputs.unsqueeze(1)
        encoder_outputs = encoder_outputs.repeat(1,energy.size(1),1,1)
    
        # apply softmax to the energys 
        allignment = self.attn_soft(energy)
        
        # create a mask like : [1,1,1,0,0,0] whos goal is to multiply the attentions of the pads with 0, rest with 1
        idxes = torch.arange(0,max_len).unsqueeze(0).long().cuda()
        #print(idxes.size())
        mask = Variable((idxes<source_lengths.unsqueeze(1)).float())
        
        # format the mask to be same size() as the attentions
        mask = mask.unsqueeze(1).unsqueeze(3).repeat(1,allignment.size(1),1,1)
        
        # apply mask
        masked = allignment * mask
        
        # now we have to rebalance the other values so they sum to 1 again
        # this is done by dividing each value by the sum of the sequence
        # calculate sums
        msum = masked.sum(-2).repeat(1,1,masked.size(2)).unsqueeze(3)
        
        # rebalance
        attentions = masked.div(msum)
        
        # now we shape the attentions to be similar to context in size
        allignment = allignment.repeat(1,1,1,encoder_outputs.size(3))

        # make context vector by element wise mul
        context = attentions * encoder_outputs
        

        context2 = torch.sum(context,2)
        
        
        return context2
    
#     def attention2(self, hidden_to_attn, encoder_outputs, source_lengths, max_len):
#         # repeat the lstm out in third dimension and the encoder outputs in second dimension so we can make a meshgrid
#         # so we can do elementwise mul for all possible combinations of h_j and s_i
#         h_j = encoder_outputs.unsqueeze(1).repeat(1, hidden_to_attn.size(1), 1, 1)
#         s_i = hidden_to_attn.unsqueeze(2).repeat(1, 1, encoder_outputs.size(1), 1)

#         # get the dot product between the two to get the energy
#         # the unsqueezes are there to emulate transposing. so we can use matmul as torch.dot doesnt accept matrices
#         energy = s_i.unsqueeze(3).matmul(h_j.unsqueeze(4)).squeeze(4)

#         #         # this is concat attention, its a different form then the ones we need
#         #         cat = torch.cat((s_i,h_j),3)

#         #         energy = self.attn_layer(cat)

#         # reshaping the encoder outputs for later
#         encoder_outputs = encoder_outputs.unsqueeze(1)
#         encoder_outputs = encoder_outputs.repeat(1, energy.size(1), 1, 1)

#         # apply softmax to the energys
#         allignment = self.attn_soft(energy)

#         # create a mask like : [1,1,1,0,0,0] whos goal is to multiply the attentions of the pads with 0, rest with 1

#         idxes = torch.arange(0, max_len).unsqueeze(0).long()

#         if torch.cuda.is_available():
#             idxes = idxes.cuda()

#         mask = Variable((idxes < source_lengths.unsqueeze(1)).float())

#         # format the mask to be same size() as the attentions
#         mask = mask.unsqueeze(1).unsqueeze(3).repeat(1, allignment.size(1), 1, 1)

#         # apply mask
#         masked = allignment * mask

#         # now we have to rebalance the other values so they sum to 1 again
#         # this is done by dividing each value by the sum of the sequence
#         # calculate sums
#         msum = masked.sum(-2).repeat(1, 1, masked.size(2)).unsqueeze(3)

#         # rebalance
#         attentions = masked.div(msum)

#         # now we shape the attentions to be similar to context in size
#         allignment = allignment.repeat(1, 1, 1, encoder_outputs.size(3))

#         # make context vector by element wise mul
#         context = attentions * encoder_outputs

#         context2 = torch.sum(context, 2)

#         return context2
    
#     def decode(self, decoder2vocab):
#         # Turn decoder output into a probabiltiy distribution over vocabulary
#         decoder2vocab_reshape = decoder2vocab.view(-1, decoder2vocab.size(2))
#         word_probs = F.softmax(decoder2vocab_reshape)
#         word_probs = word_probs.view(
#             decoder2vocab.size(0), decoder2vocab.size(1), decoder2vocab.size(2)
#         )

#         return word_probs



class Encoder(nn.Module):
    def __init__(
        self, src_emb_dim,
        src_vocab_size,
        src_hidden_dim,
        pad_token_src,
        drop,
        position_based=True,
        LSTM_instead=False,
        bidirectional=False,
        nlayers_src=1
        
    ):
        super(Encoder, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.src_emb_dim = src_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.pad_token_src = pad_token_src
        self.bidirectional = bidirectional
        self.nlayers_src = nlayers_src
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.LSTM_instead=LSTM_instead
        self.position_based=position_based
 
        
        
        # Word Embedding look-up table for the soruce language
        self.src_embedding = nn.Embedding(
            self.src_vocab_size,
            self.src_emb_dim,
            self.pad_token_src,
        )
        self.pos_embedding = nn.Embedding(
            self.src_vocab_size,
            self.src_emb_dim,
            0,
        )
#         self.scaler = nn.Linear(
#             self.src_hidden_dim,
#             self.src_emb_dim*2,
#         )
        self.scale_h0 = nn.Linear(
            self.src_emb_dim*2, self.src_hidden_dim
        )
        
        
        
        # Encoder GRU
        self.encoder = nn.GRU(
            self.src_emb_dim // 2 if self.bidirectional else self.src_emb_dim,
            self.src_hidden_dim,
            self.nlayers_src,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
    def forward(self, input_src, src_lengths, positions):
        src_emb = self.src_embedding(input_src) # BxSxE
        
        if (not self.position_based):
            src_emb = self.dropout(src_emb)
            src_emb = pack_padded_sequence(src_emb, src_lengths, batch_first=True)
            packed_output , src_h_t = self.encoder(src_emb) # out:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1) if self.bidirectional else src_h_t[-1] # BxH
            out, _ = pad_packed_sequence(packed_output, batch_first=True)
            #out = self.scaler(out)
        else:
            src_pos = self.pos_embedding(positions)
            out = torch.cat((src_pos,src_emb),2)
            out = self.dropout(out)
            hidden = torch.mean(out,1)
            h_t = self.scale_h0(hidden)
            h_t = self.relu(h_t)
            
            
        # out = BxSxH
        return out, h_t