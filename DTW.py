import librosa
import numpy as np

from nnmnkwii.util import example_file_data_sources_for_duration_model
from nnmnkwii.datasets import FileSourceDataset
from nnmnkwii.preprocessing.alignment import IterativeDTWAligner

import utils.SignalProcessingTools as spt

def DTW(path1, path2):
    data1, fs1 = spt.read_data(path1)
    data2, fs2 = spt.read_data(path2)
    assert fs1==fs2, "fs does not match"
    fs = fs1

    D1 = np.abs(librosa.stft(data1))
    fo1, sp1, ap1 = spt.get_para(data1, fs)
    mcep1 = spt.sp2mc(sp1)

    D2 = np.abs(librosa.stft(data2))
    fo2, sp2, ap2 = spt.get_para(data2, fs)
    mcep2 = spt.sp2mc(sp2)

    # D1 = FileSourceDataset(D1).asarray()
    # D2 = FileSourceDataset(D2).asarray()
    # D1_aligned, D2_aligned = IterativeDTWAligner(n_iter=1).transform((D1, D2))

    w1 = librosa.istft(D1_aligned)
    w2 = librosa.istft(D2_aligned)

    spt.save_wav(w1, fs, "./Output/jvs1_alignment")
    spt.save_wav(w2, fs, "./Output/jvs2_alignment")

if __name__ == '__main__':
    FILEPATH1 = "./Data/jvs001/VOICEACTRESS100_001.wav"
    FILEPATH2 = "./Data/jvs002/VOICEACTRESS100_001.wav"

    # DTW(FILEPATH1, FILEPATH2);