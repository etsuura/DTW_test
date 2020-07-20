import numpy as np
from numpy import fft
from scipy.io import wavfile
import math

import pysptk
from pysptk import conversion
import pyworld as pw


def read_data(path):
    fs, data = wavfile.read(path)
    data = data.astype(np.float)    # floatでないとworldは扱えない
    return data, fs

def save_wav(wav, fs, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path+".wav", fs, wav.astype(np.int16))

def get_para(data, fs):
    # This function is the same as wav2world.
    _fo, _time = pw.dio(data, fs)               # 基本周波数の抽出
    fo = pw.stonemask(data, _fo, _time, fs)     # 基本周波数の修正
    sp = pw.cheaptrick(data, fo, _time, fs)     # スペクトル包絡の抽出
    ap = pw.d4c(data, fo, _time, fs)            # 非周期性指標の抽出
    return fo, sp, ap

def path2param(path):
    fs, data = wavfile.read(path)
    data = data.astype(np.float)  # floatでないとworldは扱えない
    _fo, _time = pw.dio(data, fs)  # 基本周波数の抽出
    fo = pw.stonemask(data, _fo, _time, fs)  # 基本周波数の修正
    sp = pw.cheaptrick(data, fo, _time, fs)  # スペクトル包絡の抽出
    ap = pw.d4c(data, fo, _time, fs)  # 非周期性指標の抽出
    return fo, sp, ap

def synthesize(fo, sp, ap, fs):
    synthesized = pw.synthesize(fo, sp, ap, fs) # 音声合成
    synthesized = synthesized.astype(np.int16)  # 自動で型変換する関数作りたい
    return synthesized

def synthesize_write(filename, fo, sp, ap, fs):
    synth_voice = synthesize(fo, sp, ap, fs)
    wavfile.write(filename, fs, synth_voice)

def sp2mc(sp, order=39, alpha=0.41):   # alpha is all-pass constant
    fftlen = (len(sp) - 1) * 2
    mcep = conversion.sp2mc(sp, order, alpha)
    return mcep

def ap2bap(ap, fs):
    bap = pw.code_aperiodicity(ap, fs)
    return bap