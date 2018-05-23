import torch
import torch.nn as nn


class Tacotron(nn.Module):
    def __init__(self,
                 embedding_dim=256,
                 linear_dim=1025,
                 mel_dim=80,
                 r=5,
                 padding_idx=None):
        # r是什么意思
        super(Tacotron, self).__init__()
        self.r = r
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        # TODO symbols 是symbols文件里面的，现在还没有写
        # embedding module contains [len(symbols)] tensors of size [embedding_dim] with padding_idx
        # if padding_idx = 0 ==> padding 0
        self.embedding = nn.Embedding(
            len(symbols), embedding_dim, padding_idx=padding_idx)
        print("| > number of characted : {}".format(len(symbols)))
        self.embedding.weight.data.normal_(0, 0.3)
        # TODO encoder decoder CBHG... 网络的实现
        self.encoder = ...
        self.mel_decoder = ...
        # TODO 实际在写的时候应该要把linear和post合在一个post里面写
        self.post_decoder = ...
        self.linear_decoder = ...

    def forward(self, characters, mel_sp=None):
        B = characters.size(0)
        inputs = self.embedding(characters)
        # batch x time x dim
        encoder_ouput = self.encoder(inputs)
        # batch x time x dim x r
        mel_outputs, aligments = self.mel_decoder(encoder_ouput, mel_sp)
        # batch x time x dim
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.linear_decoder(self.post_decoder(mel_outputs))
        return mel_outputs, linear_outputs, aligments
