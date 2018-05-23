import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from hparam import tacotron_hparams
import random
from util import log
import pdb

use_cuda = torch.cuda.is_available()


class SeqLinear(nn.Module):
    """
    Linear layer for sequences
    """

    def __init__(self, input_size, output_size, time_dim=2):
        """
        :param input_size: dimension of input
        :param output_size: dimension of output
        :param time_dim: index of time dimension
        """
        super(SeqLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.time_dim = time_dim
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input_):
        """

        :param input_: sequences
        :return: outputs
        """
        batch_size = input_.size()[0]
        if self.time_dim == 2:
            input_ = input_.transpose(1, 2).contiguous()
        input_ = input_.view(-1, self.input_size)

        out = self.linear(input_).view(batch_size, -1, self.output_size)

        if self.time_dim == 2:
            out = out.contiguous().transpose(1, 2)

        return out


class Prenet(nn.Module):
    """
    Prenet before passing through the network
    """

    def __init__(self, input_size, hidden_size, output_size):
        """

        :param input_size: dimension of input
        :param hidden_depth: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(Prenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(
            OrderedDict([
                ('fc1', SeqLinear(self.input_size, self.hidden_size)),
                ('relu1', nn.ReLU()),
                ('dropout1', nn.Dropout(0.5)),
                ('fc2', SeqLinear(self.hidden_size, self.output_size)),
                ('relu2', nn.ReLU()),
                ('dropout2', nn.Dropout(0.5)),
            ]))

    def forward(self, input_):

        out = self.layer(input_)

        return out


class CBHG(nn.Module):
    """
    CBHG Module
    """

    def __init__(self,
                 hidden_depth,
                 K=16,
                 projection_size=128,
                 num_gru_layers=2,
                 max_pool_kernel_size=2,
                 is_post=False):
        """

        :param hidden_depth: dimension of hidden unit
        :param K: # of convolution banks
        :param projection_size: dimension of projection unit
        :param num_gru_layers: # of layers of GRUcell
        :param max_pool_kernel_size: max pooling kernel size
        :param is_post: whether post processing or not
        """
        super(CBHG, self).__init__()
        self.hidden_depth = hidden_depth
        self.num_gru_layers = num_gru_layers
        self.projection_size = projection_size
        self.convbank_list = nn.ModuleList()
        self.convbank_list.append(
            nn.Conv1d(
                in_channels=projection_size,
                out_channels=hidden_depth,
                kernel_size=1,
                padding=int(np.floor(1 / 2))))

        for i in range(2, K + 1):
            self.convbank_list.append(
                nn.Conv1d(
                    in_channels=hidden_depth,
                    out_channels=hidden_depth,
                    kernel_size=i,
                    padding=int(np.floor(i / 2))))

        self.batchnorm_list = nn.ModuleList()
        for i in range(1, K + 1):
            self.batchnorm_list.append(nn.BatchNorm1d(hidden_depth))

        convbank_outdim = hidden_depth * K
        if is_post:
            self.conv_projection_1 = nn.Conv1d(
                in_channels=convbank_outdim,
                out_channels=hidden_depth * 2,
                kernel_size=3,
                padding=int(np.floor(3 / 2)))
            self.conv_projection_2 = nn.Conv1d(
                in_channels=hidden_depth * 2,
                out_channels=projection_size,
                kernel_size=3,
                padding=int(np.floor(3 / 2)))
            self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_depth * 2)

        else:
            self.conv_projection_1 = nn.Conv1d(
                in_channels=convbank_outdim,
                out_channels=hidden_depth,
                kernel_size=3,
                padding=int(np.floor(3 / 2)))
            self.conv_projection_2 = nn.Conv1d(
                in_channels=hidden_depth,
                out_channels=projection_size,
                kernel_size=3,
                padding=int(np.floor(3 / 2)))
            self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_depth)

        self.batchnorm_proj_2 = nn.BatchNorm1d(projection_size)

        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        self.highway = Highwaynet(self.projection_size)
        self.gru = nn.GRU(
            self.projection_size,
            self.hidden_depth,
            num_layers=2,
            batch_first=True,
            bidirectional=True)

    def _conv_fit_dim(self, x, kernel_size=3):
        if kernel_size % 2 == 0:
            return x[:, :, :-1]
        else:
            return x

    def forward(self, input_):
        
        # pdb.set_trace() 
        input_ = input_.contiguous()
        batch_size = input_.size()[0]

        convbank_list = list()
        convbank_input = input_

        # Convolution bank filters
        for k, (conv, batchnorm) in enumerate(
                zip(self.convbank_list, self.batchnorm_list)):
            convbank_input = F.relu(
                batchnorm(
                    self._conv_fit_dim(conv(convbank_input),
                                       k + 1).contiguous()))
            convbank_list.append(convbank_input)

        # Concatenate all features
        conv_cat = torch.cat(convbank_list, dim=1)

        # Max pooling
        conv_cat = self.max_pool(conv_cat)[:, :, :-1]

        # Projection
        conv_projection = F.relu(
            self.batchnorm_proj_1(
                self._conv_fit_dim(self.conv_projection_1(conv_cat))))
        conv_projection = self.batchnorm_proj_2(
            self._conv_fit_dim(
                self.conv_projection_2(conv_projection))) + input_

        # Highway networks
        highway = self.highway.forward(conv_projection)
        highway = torch.transpose(highway, 1, 2)

        # Bidirectional GRU
        if use_cuda:
            init_gru = Variable(
                torch.zeros(2 * self.num_gru_layers, batch_size,
                            self.hidden_depth)).cuda()
        else:
            init_gru = Variable(
                torch.zeros(2 * self.num_gru_layers, batch_size,
                            self.hidden_depth))

        self.gru.flatten_parameters()
        out, _ = self.gru(highway, init_gru)

        return out


class Highwaynet(nn.Module):
    """
    Highway network
    """

    def __init__(self, num_units, num_layers=4):
        """

        :param num_units: dimension of hidden unit
        :param num_layers: # of highway layers
        """
        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.gates = nn.ModuleList()
        self.linears = nn.ModuleList()
        for _ in range(self.num_layers):
            self.linears.append(SeqLinear(num_units, num_units))
            self.gates.append(SeqLinear(num_units, num_units))

    def forward(self, input_):

        out = input_

        # highway gated function
        for fc1, fc2 in zip(self.linears, self.gates):

            h = F.relu(fc1.forward(out))
            t = F.sigmoid(fc2.forward(out))

            c = 1. - t
            out = h * t + out * c

        return out


class AttentionDecoder(nn.Module):
    """
    Decoder with attention mechanism (Vinyals et al.)
    """

    def __init__(self, num_units):
        """

        :param num_units: dimension of hidden units
        """
        super(AttentionDecoder, self).__init__()
        self.num_units = num_units

        self.v = nn.Linear(num_units, 1, bias=False)
        self.W1 = nn.Linear(num_units, num_units, bias=False)
        self.W2 = nn.Linear(num_units, num_units, bias=False)

        self.attn_grucell = nn.GRUCell(num_units // 2, num_units)
        self.gru1 = nn.GRUCell(num_units, num_units)
        self.gru2 = nn.GRUCell(num_units, num_units)

        self.attn_projection = nn.Linear(num_units * 2, num_units)
        self.out = nn.Linear(num_units, tacotron_hparams["num_mels"] *
                             tacotron_hparams["outputs_per_step"])

    def forward(self, decoder_input, memory, attn_hidden, gru1_hidden,
                gru2_hidden):
        # TODO:BUGS
        # pdb.set_trace()

        memory_len = memory.size()[1]
        batch_size = memory.size()[0]
        # log("batch_size: %", batch_size)

        # Get keys
        keys = self.W1(memory.contiguous().view(-1, self.num_units))
        keys = keys.view(-1, memory_len, self.num_units)
        # log("keys: %", keys)


        # Get hidden state (query) passed through GRUcell
        d_t = self.attn_grucell(decoder_input, attn_hidden)

        # Duplicate query with same dimension of keys for matrix operation (Speed up)
        d_t_duplicate = self.W2(d_t).unsqueeze(1).expand_as(memory)

        # Calculate attention score and get attention weights
        attn_weights = self.v(
            F.tanh(keys + d_t_duplicate).view(-1, self.num_units)).view(
                -1, memory_len, 1)
        attn_weights = attn_weights.squeeze(2)
        attn_weights = F.softmax(attn_weights)
        # log("atten_weights:%", attn_weights)

        # Concatenate with original query
        d_t_prime = torch.bmm(attn_weights.view([batch_size, 1, -1]),
                              memory).squeeze(1)

        # Residual GRU
        gru1_input = self.attn_projection(torch.cat([d_t, d_t_prime], 1))
        gru1_hidden = self.gru1(gru1_input, gru1_hidden)
        gru2_input = gru1_input + gru1_hidden

        gru2_hidden = self.gru2(gru2_input, gru2_hidden)
        bf_out = gru2_input + gru2_hidden

        # Output
        output = self.out(bf_out).view(-1, tacotron_hparams
        ["num_mels"],
                                       tacotron_hparams["outputs_per_step"])

        # return output, d_t, gru1_hidden, gru2_hidden
        return output, d_t, gru1_hidden, gru2_hidden, attn_weights

    def inithidden(self, batch_size):
        if use_cuda:
            attn_hidden = Variable(
                torch.zeros(batch_size, self.num_units),
                requires_grad=False).cuda()
            gru1_hidden = Variable(
                torch.zeros(batch_size, self.num_units),
                requires_grad=False).cuda()
            gru2_hidden = Variable(
                torch.zeros(batch_size, self.num_units),
                requires_grad=False).cuda()
        else:
            attn_hidden = Variable(
                torch.zeros(batch_size, self.num_units), requires_grad=False)
            gru1_hidden = Variable(
                torch.zeros(batch_size, self.num_units), requires_grad=False)
            gru2_hidden = Variable(
                torch.zeros(batch_size, self.num_units), requires_grad=False)

        return attn_hidden, gru1_hidden, gru2_hidden


class Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(self, embed_depth):
        """

        :param embed_depth: dimension of embedding
        """
        super(Encoder, self).__init__()
        self.embed_depth = embed_depth
        self.embed = nn.Embedding(len(tacotron_hparams["vocab"]), embed_depth)
        self.embed = nn.Embedding(len(tacotron_hparams["vocab"]), embed_depth)
        self.prenet = Prenet(embed_depth, tacotron_hparams["prenet_depths"][0],
                             tacotron_hparams["prenet_depths"][1])
        self.cbhg = CBHG(tacotron_hparams["encoder_depth"])

    def forward(self, input_):

        input_ = torch.transpose(self.embed(input_), 1, 2)
        # log("encoder input:")
        # log(input_)
        # print("encoder=>input_: %", input_)
        # print(input_)
        # pdb.set_trace() 
        prenet = self.prenet.forward(input_)
        # print("encoder=> prenet: %", prenet)
        memory = self.cbhg.forward(prenet)
        # print("encoder=>memory: %", memory)

        return memory


class MelDecoder(nn.Module):
    """
    Decoder
    """

    def __init__(self):
        super(MelDecoder, self).__init__()
        self.prenet = Prenet(tacotron_hparams["num_mels"],
                             tacotron_hparams["prenet_depths"][0],
                             tacotron_hparams["prenet_depths"][1])
        self.attn_decoder = AttentionDecoder(
            tacotron_hparams["attention_depth"])

    def forward(self, decoder_input, memory):

        # Initialize hidden state of GRUcells
        attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.inithidden(
            decoder_input.size()[0])
        outputs = list()

        # Training phase
        if self.training:
            # Prenet
            dec_input = self.prenet.forward(decoder_input)
            timesteps = dec_input.size()[2] // tacotron_hparams[
                "outputs_per_step"]

            # [GO] Frame
            prev_output = dec_input[:, :, 0]

            for i in range(timesteps):
                # TODO: have some bugs
                # pdb.set_trace()
                prev_output, attn_hidden, gru1_hidden, gru2_hidden, attn_weights = self.attn_decoder.forward(
                    prev_output,
                    memory,
                    attn_hidden=attn_hidden,
                    gru1_hidden=gru1_hidden,
                    gru2_hidden=gru2_hidden)

                outputs.append(prev_output)

                if random.random() < tacotron_hparams["teacher_forcing_ratio"]:
                    # Get spectrum at rth position
                    prev_output = dec_input[:, :, i * tacotron_hparams[
                        "outputs_per_step"]]
                else:
                    # Get last output
                    prev_output = prev_output[:, :, -1]

            # Concatenate all mel spectrogram
            outputs = torch.cat(outputs, 2)

        else:
            # [GO] Frame
            prev_output = decoder_input

            for i in range(tacotron_hparams["max_iters"]):
                prev_output = self.prenet.forward(prev_output)
                prev_output = prev_output[:, :, 0]
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.forward(
                    prev_output,
                    memory,
                    attn_hidden=attn_hidden,
                    gru1_hidden=gru1_hidden,
                    gru2_hidden=gru2_hidden)
                outputs.append(prev_output)
                prev_output = prev_output[:, :, -1].unsqueeze(2)

            outputs = torch.cat(outputs, 2)

        return outputs, attn_weights


class PostProcessingNet(nn.Module):
    """
    Post-processing Network
    """

    def __init__(self):
        super(PostProcessingNet, self).__init__()
        self.postcbhg = CBHG(
            # tacotron_hparams["hidden_depth"],
            # tacotron_hparams[""]
            # TODO ATTENTION:维度需要修改
            128,
            K=8,
            projection_size=tacotron_hparams["num_mels"],
            is_post=True)
        self.linear = SeqLinear(
            # TODO 维度需要修改
            # tacotron_hparams["hidden_depth"] * 2,
            256,
            tacotron_hparams["num_freq"])

    def forward(self, input_):
        out = self.postcbhg.forward(input_)
        out = self.linear.forward(torch.transpose(out, 1, 2))

        return out


class Tacotron(nn.Module):
    """
    End-to-end Tacotron Network
    """

    def __init__(self):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(tacotron_hparams["embed_depth"])
        self.decoder1 = MelDecoder()
        self.decoder2 = PostProcessingNet()

    def forward(self, characters, mel_input):
        # print(characters)
        # pdb.set_trace()        
        memory = self.encoder.forward(characters)
        # log("encoder finished")
        mel_output, attn_weights = self.decoder1.forward(mel_input, memory)
        # log("mel decoder finished")
        linear_output = self.decoder2.forward(mel_output)
        # log("linear decoder finished")

        return mel_output, linear_output, attn_weights