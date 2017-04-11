import librosa
from fileio import  *
import numpy as np
class Librosa:
    def __init__(self):
        fo=fileio()
        self.folder =fo.TrainingFileFolder

    def test(self):
        file_name= self.folder+ "/Record_dogstatus44.wav"
        f=open(file_name,"r")
        a=f.encoding
        print(a)
        f.close()
        X, sample_rate = librosa.load(file_name, sr=None)
        stft = np.abs(librosa.stft(X))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)


        print("x shape {0} len {1}".format(X.shape,len(X)))


        print("stft shape {0} len {1}".format(stft.shape, len(stft)))
        print("mfccs shape {0} len {1}".format(mfccs.shape, len(mfccs)))
        print("chroma shape {0} len {1}".format(chroma.shape, len(chroma)))
        print("mel shape {0} len {1}".format(mel.shape, len(mel)))
        print("contrast shape {0} len {1}".format(contrast.shape, len(contrast)))
        print("tonnetz shape {0} len {1}".format(tonnetz.shape, len(tonnetz)))
       # print(tonnetz)

if __name__== "__main__":
    lib= Librosa()
    lib.test()



