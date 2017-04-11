
import glob
import os
import librosa
import numpy as np
#import matplotlib.pyplot as plt
#import tensorflow as tf
#from matplotlib.pyplot import specgram
wavefiles=[]
def lookforwav(directory ):

    for name in os.listdir(directory) :
        fname=os.path.join(directory, name)
        if os.path.isdir(fname) :
            lookforwav(fname )
        else :
           if(  os.path.splitext(fname)[1]=='.wav'):
                 wavefiles.append(fname)
def writebacktofile(filename,wavdata,encode):
    wavdata = np.array(wavdata)
    if(encode==True):
        f = open(filename, 'w+')

        wavdata.tofile(f)

        f.close()

        # check recovery
       # f= open(filename,'r')
        #compd=np.fromfile(f,dtype=np.float32)
        #f.close()


    else :
        sr=48000
        wavdata=wavdata.flatten()
        librosa.output.write_wav(filename, wavdata, sr)

def extract_wavonly(dirname,batchsize):
    wavefiles.clear()
    wavdatabatch=[]
    batch=0
    lookforwav(dirname)

    for waves in  wavefiles:
        X, sample_rate = librosa.load(waves)

      #  X=librosa.feature.melspectrogram(X, sr=sample_rate)
        if batch==0 : wavdatabatch=[]
     #   print('file ={0} len={1}',waves ,len(X))
        if(len(X)> 48000):
       #     Y = librosa.stft(X[0:48000])
        #    Y=Y.flatten()
            #wavdatabatch.append(Y)
            wavdatabatch.append(X[0:48000])
            batch+=1
            if(batch==batchsize):
                batch=0
                yield waves,sample_rate,wavdatabatch



def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def getfeaturesforsingle(filename):
    features  = np.empty((0, 193))
    mfccs, chroma, mel, contrast, tonnetz = extract_feature(filename)
    ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    features = np.vstack([features, ext_features])
    return np.array(features)

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn[len(fn)-5])
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

if __name__ == '__main__':

        for nextbatch in extract_wavonly(os.getcwd()+'/Dataset',10):
            print(np.shape(nextbatch))

