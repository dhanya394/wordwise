import speech_recognition as sr
import matplotlib.pyplot as plt
import librosa.display
import librosa
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from numpy import array, zeros, argmin, inf, equal, ndim
from dtw import dtw

r=sr.Recognizer()
src1 = sr.AudioFile(r'E:\Major Project stuff\Major Project\Datasets\Speech Commands Dataset\seven\0a7c2a8d_nohash_0.wav')

src2 = sr.AudioFile(r'E:\Major Project stuff\Major Project\Datasets\Speech Commands Dataset\seven\0bde966a_nohash_0.wav')
with src2 as source:
    audio = r.record(source)

try:
    seven=r.recognize_sphinx(audio)
    if(seven!='seven'):
        ''' -------LOADING AUDIO AND DISPLAYING SPECTROGRAM--------- '''
        audio_path1 = r'E:\Major Project stuff\Major Project\Datasets\Speech Commands Dataset\seven\0a7c2a8d_nohash_0.wav'
        audio_path2 = r'E:\Major Project stuff\Major Project\Datasets\Speech Commands Dataset\seven\0bde966a_nohash_0.wav'
        x1, sr1 = librosa.load(audio_path1, sr=44100)
        x2, sr2 = librosa.load(audio_path2, sr=44100)
        #displaying waveform
        plt.figure(figsize=(14, 5))
        librosa.display.waveplot(x1, sr=sr1)
        plt.title("Waveform of Audio 1 ")
        plt.show()
        librosa.display.waveplot(x2, sr=sr2)
        plt.title("Waveform of Audio 2 ")
        plt.show()

        # display Spectrogram
        X1 = librosa.stft(x1)
        Xdb1 = librosa.amplitude_to_db(abs(X1))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb1, sr=sr1, x_axis='time', y_axis='hz')
        # If to pring log of frequencies
        # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar()
        plt.title("Spectrogram of Audio 1")
        plt.show()
        X2 = librosa.stft(x2)
        Xdb2 = librosa.amplitude_to_db(abs(X2))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb2, sr=sr2, x_axis='time', y_axis='hz')
        plt.colorbar()
        plt.title("Spectrogram of Audio 2")
        plt.show()

        ''' --------------FEATURE EXTRACTION--------------- '''

        # Zero Crossing Rate : the rate at which the signal changes from positive to negative or back

        x1, sr1 = librosa.load(audio_path1)
        x2, sr2 = librosa.load(audio_path2)
        # Plot the signal:
        # plt.figure(figsize=(14, 5))
        # librosa.display.waveplot(x, sr=sr)

        # Zooming in : zoom or print spectrum for 100 array columns

        n0 = 9000
        n1 = 9100
        '''
        plt.figure(figsize=(14, 5))
        plt.plot(x[n0:n1])
        plt.grid()
        #plt.show()'''

        zero_crossings1 = librosa.zero_crossings(x1[n0:n1], pad=False)
        print("Zero Crossings of Audio 1 = ", sum(zero_crossings1))
        zero_crossings2 = librosa.zero_crossings(x2[n0:n1], pad=False)
        print("Zero Crossings of Audio 1 = ", sum(zero_crossings2))
        plt.figure(figsize=(14, 5))
        plt.plot(x2[n0:n1])
        plt.grid()
        plt.title("Zero Crossing Rate")
        plt.show()

        # spectral centroid -- centre of mass -- weighted mean of the frequencies present in the sound

        spectral_centroids1 = librosa.feature.spectral_centroid(x1, sr=sr1)[0]
        #spectral_centroids.shape
        #print("Spectral Centroids = ", spectral_centroids)
        print("Spectral Centroid 1: ", spectral_centroids1)
        # Computing the time variable for visualization
        frames = range(len(spectral_centroids1))
        t = librosa.frames_to_time(frames)
        spectral_centroids2 = librosa.feature.spectral_centroid(x2, sr=sr2)[0]
        #spectral_centroids.shape
        # print("Spectral Centroids = ", spectral_centroids)
        print("Spectral Centroid 2 : ", spectral_centroids2)


        # Normalising the spectral centroid for visualisation
        def normalize(x1, axis=0):
            return sklearn.preprocessing.minmax_scale(x1, axis=axis)


        # Plotting the Spectral Centroid along the waveform
        librosa.display.waveplot(x1, sr=sr1, alpha=0.4)
        plt.plot(t, normalize(spectral_centroids1), color='r')
        plt.title("Normalized Spectral Centroids")
        plt.show()

        # Spectral Rolloff : the frequency below which a specified percentage of the total spectral energy, e.g. 85%, lies

        spectral_rolloff1 = librosa.feature.spectral_rolloff(x1, sr=sr1)[0]
        spectral_rolloff2 = librosa.feature.spectral_rolloff(x2, sr=sr2)[0]
        librosa.display.waveplot(x2, sr=sr2, alpha=0.4)
        plt.plot(t, normalize(spectral_rolloff2), color='r')
        plt.title("Spectral Rolloff")
        plt.show()
        print("Spectral Rolloff of Audio 1 : ", spectral_rolloff1)
        print("Spectral Rolloff of Audio 2 : ", spectral_rolloff2)

        # MFCC — Mel-Frequency Cepstral Coefficients
        # small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope

        mfccs1 = librosa.feature.mfcc(x1, sr=sr1)
        mfccs2 = librosa.feature.mfcc(x2, sr=sr2)

        print("MFCC of Audio 1 : ",mfccs1)
        print("MFCC of Audio 2 : ",mfccs2)
        # Displaying  the MFCCs:
        librosa.display.specshow(mfccs1, sr=sr1, x_axis='time')
        plt.show()
        librosa.display.specshow(mfccs2, sr=sr2, x_axis='time')
        plt.show()
        print(type(mfccs1))
        print(mfccs2.shape)
        plt.plot(mfccs1[0], label="MFCC1")
        plt.plot()


        #dist, cost, path = dtw(mfccs1.T, mfccs2.T)
        l2_norm = lambda mfccs1, mfccs2: (mfccs1 - mfccs2) ** 2
        dist, cost, acc_cost_matrix, path = dtw(mfccs1[0].T, mfccs2[0].T, dist=l2_norm)
        print("The normalized distance between the two : ", dist)  # 0 for similar audios
        plt.imshow(cost.T, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')
        plt.plot(path[0], path[1], 'w')  # creating plot for DTW
        plt.title("DTW")
        plt.show()  # To display the plots graphically




except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print("Could not request results; {0}".format(e))