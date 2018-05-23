import six


class Hparam(dict):
    def __init__(self, hparam=None):
        super(Hparam, self).__init__(hparam)

    def __getattr__(self, key):
        return super(Hparam, self).get(key, None)

    def __str__(self):
        tmp = ''
        for key, val in six.iteritems(self):
            if tmp:
                tmp += '\n'
            tmp += key + '=' + str(val)
        return self.__class__.__name__ + '(' + tmp + ')'

    def __setattr__(self, key, value):
        self[key] = value

    def print_hp(self):
        tmp = ''
        for key, val in six.iteritems(self):
            if tmp:
                tmp += "\n"
            tmp += key + '=' + str(val)
        print(tmp)


def get_hparam(*args, **kwargs):
    assert len(args) == 0
    return Hparam(kwargs)


tacotron_hparams = get_hparam(
    # file and path
    checkpoint_path = "./checkpoint",
    cleaners='english_cleaners',
    base_path = "./",
    data_path = "data/LJSpeech-1.1",
    log_path = './logs-tacotron',
    vocab = "abcdefghijklmnopqrstuvwxyz '.?E",


    # Audio:
    num_mels=80,
    num_freq=1025,
    sample_rate=20000,
    frame_length_ms=50,
    frame_shift_ms=12.5,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,

    # Model:
    outputs_per_step=5,
    embed_depth=256,
    prenet_depths=[256, 128],
    # encoder_depth=256,
    encoder_depth=128,
    postnet_depth=256,
    attention_depth=256,
    decoder_depth=256,

    # Training:
    batch_size=32,
    adam_beta1=0.9,
    adam_beta2=0.999,
    initial_learning_rate=0.002,
    # decay_learning_rate=True,
    decay_learning_rate=5e-4,
    # Use CMUDict during training to learn pronunciation of ARPAbet phonemes
    use_cmudict=False,  
    teacher_forcing_ratio=1.0,

    # Eval:
    epochs=50,        
    max_iters=400,
    save_step = 100,
    griffin_lim_iters=60,
    power=1.5,  # Power to raise magnitudes to prior to Griffin-Lim
)

# print(Hparam.__str__(my_hparams))
# Hparam.print_hp(my_hparams)