import librosa
import librosa.filters
import numpy as np
from numpy.linalg import norm
from scipy import signal
from scipy.io import wavfile
from scipy import interpolate
import pysptk
import pyworld as pw

from nnmnkwii.metrics import melcd
from fastdtw import fastdtw

import matplotlib.pyplot as plt
from IPython.display import Audio
import utils.SignalProcessingTools as spt

def change_size(mcep1, mcep2):
    mcep1, mcep2 = mcep1.T, mcep2.T

    m = max(mcep1.shape[-1], mcep2.shape[-1])
    mcep1 = np.pad(mcep1, ((0, 0), (0, m - mcep1.shape[-1])), mode='edge')
    mcep2 = np.pad(mcep2, ((0, 0), (0, m - mcep2.shape[-1])), mode='edge')
    return mcep1, mcep2

def plot_para(m, name, T=False):
    idx = 0
    plt.figure(figsize=(16, 8))
    if T:
        plt.imshow(m.T[::-1])
    else:
        plt.imshow(m[::-1])
    plt.savefig(name+".png".format(idx), bbox_inches="tight", pad_inches=0.0)

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


def alignment(X, Y, param1, param2):
    idx = 0
    param1_a, param2_a = DTWAligner(verbose=0, dist=melcd).transform((X, Y), (param1.T[None], param2.T[None]))
    param1_a = np.squeeze(param1_a)
    param2_a = np.squeeze(param2_a)
    param1_a, param2_a = param1_a.T, param2_a.T
    return param1_a, param2_a

def alignment_param(path1, path2, name1, name2):
    # classにする？
    fo1, sp1, ap1 = spt.path2param(path1)
    fo2, sp2, ap2 = spt.path2param(path2)
    mcp1 = spt.sp2mc(sp1)
    mcp2 = spt.sp2mc(sp2)
    mcp1, mcp2 = change_size(mcp1, mcp2)

    _, fs = spt.read_data(path1)
    bap1 = spt.ap2bap(ap1, fs)
    bap2 = spt.ap2bap(ap2, fs)
    bap1, bap2 = change_size(bap1, bap2)

    X, Y = mcp1.T[None], mcp2.T[None]

    # mcp1_alig, mcp2_alig = alignment(X, Y, mcp1, mcp2)
    # plot_para(mcp1, "mcp1")
    # plot_para(mcp2, "mcp2")
    # plot_para(mcp1_alig, "mcp1_alig")
    # plot_para(mcp2_alig, "mcp2_alig")

    # bap1_alig, bap2_alig = alignment(X, Y, bap1, bap2)
    # plot_para(bap1, "bap1")
    # plot_para(bap2, "bap2")
    # plot_para(bap1_alig, "bap1_alig")
    # plot_para(bap2_alig, "bap2_alig")

    fo1, fo2 = fo1


    pass

def main():
    FILEPATH1 = "./Data/jvs001/VOICEACTRESS100_001.wav"
    FILEPATH2 = "./Data/jvs002/VOICEACTRESS100_001.wav"
    NV_PATH = "./Data/Audio/ns100.007.wav"
    EL_PATH = "./Data/Audio/ne100.007.wav"
    EU_PATH = "./Data/Audio/seou_eu.wav"

    # alignment_param(FILEPATH1, FILEPATH2, "EL_alig", "NV_alig")
    alignment_param(NV_PATH, EL_PATH, "EL_alig", "NV_alig")

    pass

if __name__ == '__main__':
    main()