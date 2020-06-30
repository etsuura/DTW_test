# reference https://blog.hiroshiba.jp/sandbox-alignment-voice-actress-data/

import librosa
import librosa.filters
import numpy as np
from numpy.linalg import norm
from scipy import signal
from scipy.io import wavfile
from scipy import interpolate
import pysptk
import pyworld

from nnmnkwii.metrics import melcd
from fastdtw import fastdtw

import matplotlib.pyplot as plt
from IPython.display import Audio

import utils.SignalProcessingTools as spt

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def save_wav(wav, fs, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, fs, wav.astype(np.int16))

# プリエンファシスフィルタ 高域成分を強調する
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html#scipy.signal.lfilter
def preemphasis(x, hparams):
    return signal.lfilter([1, -hparams.preemphasis], [1], x)

def inv_preemphasis(x, hparams):
    return signal.lfilter([1], [1, -hparams.preemphasis], x)

def spectrogram(y, hparams):
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    return _normalize(S)

def inv_spectrogram(spectrogram, hparams):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)    # Convert back to linear
    return inv_preemphasis(_griffin_lim(S ** hparams.power))                    # Reconstruct phase

def melspectrogram(y, hparams):
    D = _stft(preemphasis(y, hparams), hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams))
    return _normalize(S, hparams)

def find_endpoint(wav, hparams, threshold_db=-40, min_silence_sec=0.8):
    window_length = int(hparams.sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x+window_length]) < threshold:
            return x + hop_length
    return len(wav)

def _griffin_lim(S, hparams):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y

def _stft(y, hparams):
    n_fft, hop_length, win_length = _stft_parameters(hparams)
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _istft(y, hparams):
    _, hop_length, win_length = _stft_parameters(hparams)
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)

def _stft_parameters(hparams):
    # check hparams
    # n_fft = (hparams.num_freq - 1) * 2
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    n_fft = win_length
    return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectrogram)

def _build_mel_basis(hparams):
    # n_fft = (hparams.num_freq - 1) * 2
    n_fft = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def _normalize(S, hparams):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)

def _denormalize(S, hparams):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db


# from https://github.com/r9y9/nnmnkwii/blob/4cade86b5c35b4e35615a2a8162ddc638018af0e/nnmnkwii/preprocessing/alignment.py#L14
class DTWAligner(object):
    def __init__(self, dist=lambda x, y: norm(x - y), radius=1, verbose=0):
        self.verbose = verbose
        self.dist = dist
        self.radius = radius

    def transform(self, XY_src, XY_dst=None):
        if XY_dst is None:
            XY_dst = XY_src

        X_src, Y_src = XY_src
        X_dst, Y_dst = XY_dst
        assert X_src.ndim == 3 and Y_src.ndim == 3
        assert X_dst.ndim == 3 and Y_dst.ndim == 3

        longer_features = X_dst if X_dst.shape[1] > Y_dst.shape[1] else Y_dst

        X_aligned = np.zeros_like(longer_features)
        Y_aligned = np.zeros_like(longer_features)
        for idx, (x_src, y_src, x_dst, y_dst) in enumerate(zip(X_src, Y_src, X_dst, Y_dst)):
            dist, path = fastdtw(x_src, y_src, radius=self.radius, dist=self.dist)
            dist /= (len(x_src) + len(y_src))

            pathx = np.array(list(map(lambda l: l[0], path))) / len(x_src)
            pathx = interpolate.interp1d(np.linspace(0, 1, len(pathx)), pathx)(np.linspace(0, 1, len(x_dst)))
            pathx = np.floor(pathx * len(x_dst)).astype(np.int)

            pathy = np.array(list(map(lambda l: l[1], path))) / len(y_src)
            pathy = interpolate.interp1d(np.linspace(0, 1, len(pathy)), pathy)(np.linspace(0, 1, len(y_dst)))
            pathy = np.floor(pathy * len(y_dst)).astype(np.int)

            x_dst, y_dst = x_dst[pathx], y_dst[pathy]
            max_len = max(len(x_dst), len(y_dst))
            if max_len > X_aligned.shape[1] or max_len > Y_aligned.shape[1]:
                pad_size = max(max_len - X_aligned.shape[1],
                               max_len > Y_aligned.shape[1])
                X_aligned = np.pad(
                    X_aligned, [(0, 0), (0, pad_size), (0, 0)],
                    mode="constant", constant_values=0)
                Y_aligned = np.pad(
                    Y_aligned, [(0, 0), (0, pad_size), (0, 0)],
                    mode="constant", constant_values=0)
            X_aligned[idx][:len(x_dst)] = x_dst
            Y_aligned[idx][:len(y_dst)] = y_dst
            if self.verbose > 0:
                print("{}, distance: {}".format(idx, dist))
        return X_aligned, Y_aligned

# from https://github.com/r9y9/nnmnkwii/blob/8afc05cce5b8a6727ed5d0fb874c1ae4e4039f1e/tests/test_real_datasets.py#L113
# fs = hparams.sample_rate
# fftlen = pyworld.get_cheaptrick_fft_size(fs)
# alpha = pysptk.util.mcepalpha(fs)
# order = 25
# frame_period = 5
# hop_length = int(fs * (frame_period * 0.001))

def collect_features(x, fs):
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    alpha = pysptk.util.mcepalpha(fs)
    order = 25
    frame_period = 5
    hop_length = int(fs * (frame_period * 0.001))

    x = x.astype(np.float64)
    _f0, _timeaxis = pyworld.dio(x, fs, frame_period=frame_period)
    f0 = pyworld.stonemask(x, _f0, _timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, _timeaxis, fs)
    mc = pysptk.sp2mc(spectrogram, order=order, alpha=alpha)
    return mc

# from https://github.com/keithito/tacotron/blob/08989cc3553b3a916a31f565e4f20e34bf19172f/hparams.py
def calc_hparams(fs):
    hparams = AttrDict(
        # Audio:
        num_mels=80,
        # num_freq=2049,
        # n_fft = 2048,
        sample_rate=fs,
        frame_length_ms=50,
        frame_shift_ms=12.5,
        # frame_shift_ms=25,
        preemphasis=0.97,
        min_level_db=-100,
        ref_level_db=20,
    )

    return hparams

def calc(base1, base2):
    feature1 = []
    feature2 = []
    mel1 = []
    mel2 = []
    spec1 = []
    spec2 = []

    for i in range(1):
        # p = glob.glob(os.path.join(base1, '*{0:03}*'.format(i + 1)))[0]
        # w = load_wav(base1)
        w1, fs1 = spt.read_data(base1)
        w2, fs2 = spt.read_data(base2)

        assert fs1 == fs2, "fs does not match"
        hparams = calc_hparams(fs1)

        m1 = melspectrogram(w1, hparams).astype(np.float32)
        s1 = _stft(w1, hparams)
        f1 = collect_features(w1, fs1).T

        # p = glob.glob(os.path.join(base2, '*{0:03}*'.format(i + 1)))[0]
        # w = load_wav(base2)
        m2 = melspectrogram(w2, hparams).astype(np.float32)
        s2 = _stft(w2, hparams)
        f2 = collect_features(w2, fs2).T

        m = max(f1.shape[-1], f2.shape[-1])
        f1 = np.pad(f1, ((0, 0), (0, m - f1.shape[-1])), mode='edge')   #padding shapeが大きい方にサイズを揃える
        f2 = np.pad(f2, ((0, 0), (0, m - f2.shape[-1])), mode='edge')

        m = max(s1.shape[-1], s2.shape[-1])
        s1 = np.pad(s1, ((0, 0), (0, m - s1.shape[-1])), mode='edge')
        s2 = np.pad(s2, ((0, 0), (0, m - s2.shape[-1])), mode='edge')

        m = max(m1.shape[-1], m2.shape[-1])
        m1 = np.pad(m1, ((0, 0), (0, m - m1.shape[-1])), mode='edge')
        m2 = np.pad(m2, ((0, 0), (0, m - m2.shape[-1])), mode='edge')

        feature1.append(f1)
        feature2.append(f2)
        spec1.append(s1)
        spec2.append(s2)
        mel1.append(m1)
        mel2.append(m2)

    return feature1, feature2, spec1, spec2, mel1, mel2, fs1




def plot_mel(m, name, T=False):
    idx = 0
    plt.figure(figsize=(16, 8))
    if T:
        plt.imshow(m[0].T[::-1])
    else:
        plt.imshow(m[0][::-1])
    plt.savefig(name+".png".format(idx), bbox_inches="tight", pad_inches=0.0)

def alignment(feature1, feature2, spec1, spec2, mel1, mel2):
    idx = 0
    X, Y = feature1[idx].T[None], feature2[idx].T[None]
    spec1_aligned, spec2_aligned = DTWAligner(verbose=0, dist=melcd).transform((X, Y), (spec1[idx].T[None], spec2[idx].T[None]))
    mel1_aligned, mel2_aligned = DTWAligner(verbose=0, dist=melcd).transform((X, Y), (mel1[idx].T[None], mel2[idx].T[None]))
    # return list(X, Y, spec1_aligned, spec2_aligned, mel1_aligned, mel2_aligned)
    return X, Y, spec1_aligned, spec2_aligned, mel1_aligned, mel2_aligned

def save_alignment_voice(s, fs, name):
    idx = 0
    hparams = calc_hparams(fs)
    w = _istft(s[0].T, hparams)
    save_wav(w, fs, name+'.wav'.format(idx))
    # Audio(w, rate=hparams.sample_rate)

def alignment_audio(path1, path2, name1="name1", name2="name2"):
    feature1, feature2, spec1, spec2, mel1, mel2, fs = calc(path1, path2)
    X, Y, spec1_aligned, spec2_aligned, mel1_aligned, mel2_aligned = alignment(feature1, feature2, spec1, spec2, mel1, mel2)
    save_alignment_voice(spec1_aligned, fs, name1)
    save_alignment_voice(spec2_aligned, fs, name2)
    pass

def stft_test(path):
    w1, fs = spt.read_data(path)
    hparams = calc_hparams(fs)
    s1 = _stft(w1, hparams)
    w = _istft(s1, hparams)
    save_wav(w, fs, "stft_test.wav")

if __name__ == '__main__':
    FILEPATH1 = "./Data/jvs001/VOICEACTRESS100_001.wav"
    FILEPATH2 = "./Data/jvs002/VOICEACTRESS100_001.wav"
    # alignment_audio(FILEPATH1, FILEPATH2, "jvs1", "jvs2")

    NV_PATH = "./Data/Audio/ns100.007.wav"
    EL_PATH = "./Data/Audio/ne100.007.wav"
    EU_PATH = "./Data/Audio/seou_eu.wav"
    alignment_audio(EL_PATH, NV_PATH, "EL_alig", "NV_alig")
    # stft_test(NV_PATH)

    # feature1, feature2, spec1, spec2, mel1, mel2 = calc(FILEPATH1, FILEPATH2)
    # plot_mel(mel1, "mel1")
    # plot_mel(mel2, "mel2")
    #
    # X, Y, spec1_aligned, spec2_aligned, mel1_aligned, mel2_aligned = alignment(feature1, feature2, spec1, spec2, mel1, mel2)
    # plot_mel(mel1_aligned, "mel1_alig", T=1)
    # plot_mel(mel2_aligned, "mel2_alig", T=1)
    # save_alignment_voice(spec1_aligned, "spec1_aligned")
    # save_alignment_voice(spec2_aligned, "spec2_aligned")