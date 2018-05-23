#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 23:11:12 2018

@author: qiu
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import os
import time
import math
import numpy as np
import librosa
import argparse
import traceback
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# import util
from data import get_dataset, fetch_batch
from util import log, init_log
from hparam import Hparam as hp
from hparam import tacotron_hparams
from nn_module import *

use_cuda = torch.cuda.is_available()


def get_git_commit():
    subprocess.check_output(['git', 'diff-index', '--quiet',
                             'HEAD'])  # Verify client is clean
    commit = subprocess.check_output(['git', 'rev-parse',
                                      'HEAD']).decode().strip()[:10]
    log('Git commit: %s' % commit)
    return commit


# 加载模型
# def load_status(model):


def curr_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def train(log_dir, args):
    # commit = get_git_commit() if args.git else 'None'
    # checkpoint_path需要修改
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
    input_path = os.path.join(args.base_dir, args.input)
    log('Checkpoint path: %s' % checkpoint_path)
    log('Loading training data from: %s' % input_path)
    log('Using model: %s' % args.model)
    log('=======================')
    log(hp.__str__(tacotron_hparams))
    log('=======================')

    if use_cuda:
        model = nn.DataParallel(Tacotron().cuda())
        # model = nn.DistributedDataParallel(Tacotron().cuda())
    else:
        model = Tacotron()

    optimizer = optim.Adam(
        model.parameters(),
        lr=tacotron_hparams["initial_learning_rate"],
        weight_decay=tacotron_hparams["decay_learning_rate"])

    step = 0

    try:
        checkpoint = torch.load(
            os.path.join(tacotron_hparams["checkpoint_path"],
                         'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log("\n--------model restored at step %d--------\n" %
            args.restore_step)

    except:
        log("\nNew Model\n")

    # Training
    model = model.train()
    train_data = get_dataset()
    # lj_data = data.get_dataset()

    # Make checkpoint directory if not exists
    if not os.path.exists(tacotron_hparams["checkpoint_path"]):
        os.mkdir(tacotron_hparams["checkpoint_path"])

    # Decide loss function
    if use_cuda:
        criterion = nn.L1Loss().cuda()
    else:
        criterion = nn.L1Loss()

    # Loss for frequency of human register ???这一步是干嘛的？？？
    n_priority_freq = int(
        3000 /
        (tacotron_hparams["sample_rate"] * 0.5) * tacotron_hparams["num_freq"])

    for epoch in range(tacotron_hparams["epochs"]):
        # 加载数据
        data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=tacotron_hparams["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=fetch_batch)
        log("load batch")

        for index, data in enumerate(data_loader):

            # log(index)
            # log(data)
            cur_step = index + args.restore_step + epoch * len(data_loader) + 1
            optimizer.zero_grad()

            try:
                mel_input = np.concatenate(
                    (np.zeros(
                        [
                            tacotron_hparams["batch_size"],
                            tacotron_hparams["num_mels"], 1
                        ],
                        dtype=np.float32), data[2][:, :, 1:]),
                    axis=2)
            except:
                log("dimension error -1")
                raise TypeError("dimension error")
            if use_cuda:
                characters = torch.from_numpy(data[0]).type(
                    torch.cuda.LongTensor).cuda()
                mel_input = torch.from_numpy(mel_input).type(
                    torch.cuda.FloatTensor).cuda()
                mel_spectrogram = torch.from_numpy(data[2]).type(
                    torch.cuda.FloatTensor).cuda()
                linear_spectrogram = torch.from_numpy(data[1]).type(
                    torch.cuda.FloatTensor).cuda()

            else:
                characters = torch.from_numpy(data[0]).type(
                    torch.cuda.LongTensor)
                mel_input = torch.from_numpy(mel_input).type(
                    torch.cuda.FloatTensor)
                mel_spectrogram = torch.from_numpy(data[2]).type(
                    torch.cuda.FloatTensor)
                linear_spectrogram = torch.from_numpy(data[1]).type(
                    torch.cuda.FloatTensor)

            mel_output, linear_output, attn_weights = model.forward(
                characters, mel_input)

            # print('attn size:{}'.format(attn_weights.size()))
            # print('attention weights:{}'.format(attn_weights))
            # Calculate loss
            mel_loss = criterion(mel_output, mel_spectrogram)
            linear_loss = torch.abs(linear_output - linear_spectrogram)
            linear_loss = 0.5 * torch.mean(linear_loss) + 0.5 * torch.mean(
                linear_loss[:, :n_priority_freq, :])
            loss = mel_loss + linear_loss
            loss = loss.cuda()

            start_time = time.time()

            # Calculate gradients
            loss.backward()

            # clipping gradients
            nn.utils.clip_grad_norm(model.parameters(), 1.)

            # Update weights
            optimizer.step()

            time_per_step = time.time() - start_time

            if cur_step % tacotron_hparams["outputs_per_step"] == 0:
                log("tps:{:.2f}  At timestep {} linear loss: {:.4f} mel loss:{:.4f} total loss:{:.4f}".format(time_per_step, cur_step, linear_loss.data[0],mel_loss.data[0], loss.data[0]))
                # log("At timestep %d" % cur_step)
                # log("linear loss: %.4f" % linear_loss.data[0])
                # log("mel loss: %.4f" % mel_loss.data[0])
                # log("total loss: %.4f" % loss.data[0])

            if cur_step % tacotron_hparams["save_step"] == 0:
                save_checkpoint(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    os.path.join(tacotron_hparams["checkpoint_path"],
                                 'checkpoint_%d.pth.tar' % cur_step))
                log("save model at step %d ..." % cur_step)
                # plot_alignment(attn_weights, cur_step)
                # AttentionDecoder.parameters
                # attn = attn.data.cpu().numpy()[0]
                # plt.imshow(attn.T, cmap='hot', interpolation='nearest')
                # plt.xlabel('Decoder Steps')
                # plt.ylabel('Encoder Steps')
                # fig_path = os.path.join(log_dir, 'attn/epoch{}.jpg'.format(epoch))
                # plt.savefig(fig_path, format='png')

                # wav = spectrogram2wav(mag_hat)
                # write(os.path.join(log_dir, 'wav/epoch{}_{}.wav'.format(epoch, i)), hp.sr, wav)
                # msg = 'synthesis {}.wav in epoch{} model'.format(i, epoch)
                # print(msg)
                # f.write(msg)

            # if current_step in hp.decay_step:
            #     optimizer = adjust_learning_rate(optimizer, current_step)


def plot_alignment(alignment, gs):
    """Plots the alignment
    alignments: A list of (numpy) matrix of shape (encoder_steps, decoder_steps)
    gs : (int) global step
    """
    fig, ax = plt.subplots()
    im = ax.imshow(alignment)
    # plt.plot(x_points, y_points, 'o', label='Input Data')
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig(
        '{}/alignment_{}k.png'.format(tacotron_hparams["logdir"],
                                      gs // tacotron_hparams["save_step"]),
        format='png')


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


# def adjust_learning_rate(optimizer, step):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     if step == 500000:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 0.0005

#     elif step == 1000000:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 0.0003

#     elif step == 2000000:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 0.0001

#     return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('./'))
    parser.add_argument('--input', default='training/train.txt')
    parser.add_argument('--model', default='tacotron')
    parser.add_argument(
        '--name',
        help='Name of the run. Used for logging. Defaults to model name.')
    parser.add_argument(
        '--hparams',
        default='',
        help=
        'Hyperparameter overrides as a comma-separated list of name=value pairs'
    )
    parser.add_argument(
        '--restore_step',
        type=int,
        help='Global step to restore from checkpoint.',
        default=0)
    parser.add_argument(
        '--summary_interval',
        type=int,
        default=100,
        help='Steps between running summary ops.')
    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        default=1000,
        help='Steps between writing checkpoints.')
    parser.add_argument(
        '--slack_url', help='Slack webhook URL to get periodic reports.')
    # parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    parser.add_argument(
        '--git',
        action='store_true',
        help='If set, verify that the client is clean.')
    args = parser.parse_args()
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
    os.makedirs(log_dir, exist_ok=True)
    init_log(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
    # args还需要修改
    # print(args.hparams)
    # print(type(args.hparams))

    # print(tacotron_hparams["decay_learning_rate"])
    # print("========================================")

    train(log_dir, args)


if __name__ == '__main__':
    main()
