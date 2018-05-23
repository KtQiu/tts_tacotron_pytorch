import atexit
from datetime import datetime
import json
from threading import Thread
from urllib.request import Request, urlopen
import numpy as np
import librosa
from hparam import tacotron_hparams
from scipy import signal


_format = '%Y-%m-%d %H:%M:%S.%f'
_run_name = None
_slack_url = None


log_file = None

def init_log(filename, run_name, slack_url=None):
  global log_file, _run_name, _slack_url
  close_logfile()
  log_file = open(filename, 'a')
  log_file.write('\n-----------------------------------------------------------------\n')
  log_file.write('Starting new training run\n')
  log_file.write('-----------------------------------------------------------------\n')
  _run_name = run_name
  _slack_url = slack_url


# 写入log
def log(msg, slack=False):
    print(msg)
    if log_file is not None:
        log_file.write('[%s]  %s\n' % (datetime.now().strftime(_format)[:-3], msg))
    # if slack and _slack_url is not None:
    #     Thread(target=_send_slack, args=(msg,)).start()



def close_logfile():
  global log_file
  if log_file is not None:
    log_file.close()
    log_file = None


# the codes below are refered from https://github.com/keithito/tacotron

_mel_basis = None

def save_wav(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  librosa.output.write_wav(path, wav.astype(np.int16), tacotron_hparams["sample_rate"])


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
    n_fft = (tacotron_hparams["num_freq"] - 1) * 2
    return librosa.filters.mel(tacotron_hparams["sample_rate"], n_fft, n_mels=tacotron_hparams["num_mels"])

def _normalize(S):
    return np.clip((S - tacotron_hparams["min_level_db"]) / -tacotron_hparams["min_level_db"], 0, 1)

def _denormalize(S):
    return (np.clip(S, 0, 1) * -tacotron_hparams["min_level_db"]) + tacotron_hparams["min_level_db"]

def _stft_parameters():
    n_fft = (tacotron_hparams["num_freq"]-1) * 2
    hop_length = int(tacotron_hparams["frame_shift_ms"] / 1000 * tacotron_hparams["sample_rate"])
    win_length = int(tacotron_hparams["frame_length_ms"] / 1000 * tacotron_hparams["sample_rate"])
    return n_fft, hop_length, win_length

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def preemphasis(x):
    return signal.lfilter([1, - tacotron_hparams["preemphasis"]], [1], x)


def inv_preemphasis(x):
    return signal.lfilter([1], [1, -tacotron_hparams["preemphasis"]], x)


def spectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - tacotron_hparams["ref_level_db"]
    return _normalize(S)


def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''

    S = _denormalize(spectrogram)
    S = _db_to_amp(S + tacotron_hparams["ref_level_db"])  # Convert back to linear

    return inv_preemphasis(_griffin_lim(S ** tacotron_hparams["power"]))          # Reconstruct phase

def _griffin_lim(S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(ref_level_db["griffin_lim_iters"]):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y

def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def melspectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D)))
    return _normalize(S)

def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(tacotron_hparams["sample_rate"]  * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)

def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, tacotron_hparams["outputs_per_step"]  - (timesteps % tacotron_hparams["outputs_per_step"] )]], mode='constant', constant_values=0.0)

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params
