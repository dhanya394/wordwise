'''from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import matplotlib.pyplot as plt
[Fs, x] = audioBasicIO.read_audio_file(r"E:\Major Project stuff\Major Project\Datasets\TORGO Dataset\Without Dysarthria\MC01\Session1\wav_headMic\0001.wav")
F, f_names = pyAudioAnalysis.ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0])
plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()'''


''' -------LOADING AUDIO AND DISPLAYING SPECTROGRAM--------- '''
import librosa
audio_path = r"E:\Major Project stuff\Major Project\Datasets\TORGO Dataset\Without Dysarthria\MC01\Session1\wav_headMic\0001.wav"
x , sr = librosa.load(audio_path, sr=44100)
print(x, sr)
from playsound import playsound
#playsound(audio_path)

#display waveform of the audio
import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
plt.title("Displaying the waveform of the audio")
plt.show()

#display Spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
#If to pring log of frequencies
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.title("Spectrogram")
plt.show()

''' --------------FEATURE EXTRACTION--------------- '''

#Zero Crossing Rate : the rate at which the signal changes from positive to negative or back

x, sr = librosa.load(audio_path)
#Plot the signal:
#plt.figure(figsize=(14, 5))
#librosa.display.waveplot(x, sr=sr)

#Zooming in : zoom or print spectrum for 100 array columns

n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()
plt.title("Zero Crossing Rate")
plt.show()

zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print("Zero Crossings = ",sum(zero_crossings))

#spectral centroid -- centre of mass -- weighted mean of the frequencies present in the sound
import sklearn
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape
print("Spectral Centroids = ",spectral_centroids)
print("Shape : ",spectral_centroids.shape)
# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')
plt.title("Normalized Spectral Centroids")
plt.show()

#Spectral Rolloff : the frequency below which a specified percentage of the total spectral energy, e.g. 85%, lies

spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
plt.title("Spectral Rolloff")
plt.show()

#MFCC — Mel-Frequency Cepstral Coefficients
#small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope

mfccs = librosa.feature.mfcc(x, sr=sr)
print(mfccs.shape)#Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.title("Mel Frequency Cepstral Coefficients")
plt.show()

'''
pitches, magnitudes = librosa.piptrack(y=x, sr=sr)
plt.imshow(pitches[:100, :], aspect="auto", interpolation="nearest", origin="bottom")
plt.title('pitches')
plt.show()'''









