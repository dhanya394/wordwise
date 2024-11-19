import scipy.io.wavfile as wavefile
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack as fft
import numpy as np
""" FAST FOURIER TRANSFORM """
rate,data = wavefile.read(r"E:\Major Project stuff\Major Project\Datasets\TORGO Dataset\Without Dysarthria\MC01\Session1\wav_headMic\0001.wav")

""" SAMPLING FREQUENCY : 
In signal processing, sampling is the reduction of a continuous-time signal to a discrete-time signal. 
A sample is a value or set of values at a point in time and/or space.
A sampler is a subsystem or operation that extracts samples from a continuous signal. 
"""
print("Frequency Sampling : ", rate)

""" CHANNELS IN AN AUDIO FILE : """
channels = len(data.shape)
print("Channels : " , channels)

if channels == 2:
    data = data.sum(axis=1) / 2
N = data.shape[0]
print("Complete Samplings N : ", N)

secs = N / float(rate)
print ("secs : ", secs)
Ts = 1.0/rate # sampling interval in time

print ("Timestep between samples Ts", Ts)
t = np.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
FFT = abs(scipy.fft.fft(data))
FFT_side = FFT[range(N//4)] # one side FFT range
freqs = scipy.fftpack.fftfreq(data.size, t[1]-t[0])
fft_freqs = np.array(freqs)
freqs_side = freqs[range(N//4)] # one side frequency range
fft_freqs_side = np.array(freqs_side)

print (abs(FFT_side))

plt.subplot(211)
p1 = plt.plot(t, data, "g") # plotting the signal
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(212)
p2 = plt.plot(freqs_side, abs(FFT_side), "b") # plotting the positive fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count single-sided')
plt.show()

