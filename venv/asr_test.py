from keras.models import load_model
model=load_model('E:/Major Project stuff/ASR/model/best_model2.hdf5')

import os
import librosa
#import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import warnings
warnings.filterwarnings("ignore")

classes = ['bed', 'cat', 'dog', 'down', 'eight', 'nine', 'no', 'seven', 'up', 'yes']

def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]


import sounddevice as sd
import soundfile as sf

samplerate = 16000
duration = 1 # seconds
filename = 'voice1.wav'
print("start")
mydata = sd.rec(int(samplerate * duration), samplerate=16000,
    channels=1, blocking=True)
print("end")
sd.wait()
sf.write(filename, mydata, samplerate)

filepath = 'C:/Users/ASUS/PycharmProjects/Speech_Assessment_Tool/venv'
samples, sample_rate = librosa.load(filepath + '/' + 'voice1.wav', sr = 16000)
samples = librosa.resample(samples, sample_rate, 8000)
#ipd.Audio(samples,rate=8000)

print("Text:",predict(samples))

