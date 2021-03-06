import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptive import *

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, adasoft=True, adainp = True,
    cutoffs = [2000,10000]):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)

        self.adainp = adainp
        if self.adainp:
            self.encoder = AdaptiveInput(ninp, ntoken, cutoffs)
        else:
            self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462

        self.adaptive_softmax = adasoft

        if self.adaptive_softmax:
            self.out = AdaptiveSoftmax(ninp, ntoken, cutoffs)
        else:
            self.decoder = nn.Linear(nhid, ntoken)
            if tie_weights:
                if nhid != ninp:
                    raise ValueError('When using the tied flag, nhid must be equal to emsize')
                self.decoder.weight = self.encoder.weight
                self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        if not self.adainp:
            nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        if not self.adaptive_softmax:
            nn.init.zeros_(self.decoder.weight)
            nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden, target):
        input_size = list(input.size())
        if self.adainp:
            input = input.view(-1) #converting to 1d (e.g., [35,20]-> [35*20])
        encoded = self.encoder(input)
        if self.adainp:
            encoded = encoded.view(input_size[0], input_size[1], -1) #converting to 3d (e.g., [35*20*200]-> [35,20,200])
        emb = self.drop(encoded)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)

        if self.adaptive_softmax:
            output = output.view(-1, output.size(2))
            linear = self.out(output,target)
        else:
            decoded = self.decoder(output)
            decoded = decoded.view(-1, self.ntoken)
            linear = F.log_softmax(decoded, dim=1)

        return linear, hidden

    def log_prob(self, input, hidden):
        embed = self.embedding(input)
        output, hidden = self.rnn(embed, hidden)
        decoded = self.decoder.log_prob(output.contiguous() \
                .view(output.size(0) * output.size(1), output.size(2)))

        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, adasoft = True, adainp = True, cutoffs = [2000,10000]):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.adainp = adainp
        if self.adainp:
            self.encoder = AdaptiveInput(ninp, ntoken, cutoffs)
        else:
            self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp

        self.adaptive_softmax = adasoft

        if self.adaptive_softmax:
            self.out = AdaptiveSoftmax(ninp, ntoken, cutoffs)
        else:
            self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        if not self.adainp:
            nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        if not self.adaptive_softmax:
            nn.init.zeros_(self.decoder.weight)
            nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, target, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src_size = list(src.size())
        if self.adainp:
            src = src.view(-1) #converting to 1d (e.g., [35,20]-> [35*20])
        src = self.encoder(src)
        if self.adainp:
            src = src.view(src_size[0], src_size[1], -1)
        else:
            src = src * math.sqrt(self.ninp)

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        if self.adaptive_softmax:
            output=output.view(-1,output.size()[2])
            linear = self.out(output,target)
        else:
            output = self.decoder(output)
            linear = F.log_softmax(output, dim=-1)
        return linear
