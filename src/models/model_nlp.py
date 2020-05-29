import torch
import torch.nn as nn
import math
import constants

#文本建模的GRU模型
class LMModel(nn.Module):
    def __init__(self, nvoc, ninput, nhid, nlayers):
        super(LMModel, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(nvoc, ninput)

        self.ninput = ninput
        self.nhid = nhid
        self.nlayers = nlayers
        self.rnn = nn.GRU(input_size = ninput, hidden_size = nhid, num_layers = nlayers)
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()

        

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)     

    def forward(self, input):
        embeddings = self.drop(self.encoder(input))
        output, hidden = self.rnn(embeddings)        
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
    
