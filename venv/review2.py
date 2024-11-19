import speech_recognition as sr
import matplotlib.pyplot as plt
import librosa.display
import librosa
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from numpy import array, zeros, argmin, inf, equal, ndim

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
        plt.plot(x1[n0:n1])
        plt.grid()
        plt.title("Zero Crossing Rate of Audio 1")
        plt.show()
        plt.figure(figsize=(14, 5))
        plt.plot(x2[n0:n1])
        plt.grid()
        plt.title("Zero Crossing Rate of Audio 2")
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
        plt.title("Normalized Spectral Centroids of Audio 1")
        plt.show()
        librosa.display.waveplot(x2, sr=sr2, alpha=0.4)
        plt.plot(t, normalize(spectral_centroids1), color='r')
        plt.title("Normalized Spectral Centroids of Audio 2")
        plt.show()

        # Spectral Rolloff : the frequency below which a specified percentage of the total spectral energy, e.g. 85%, lies

        spectral_rolloff1 = librosa.feature.spectral_rolloff(x1, sr=sr1)[0]
        spectral_rolloff2 = librosa.feature.spectral_rolloff(x2, sr=sr2)[0]
        librosa.display.waveplot(x1, sr=sr1, alpha=0.4)
        plt.plot(t, normalize(spectral_rolloff1), color='r')
        plt.title("Spectral Rolloff of Audio 1")
        plt.show()
        librosa.display.waveplot(x2, sr=sr2, alpha=0.4)
        plt.plot(t, normalize(spectral_rolloff2), color='r')
        plt.title("Spectral Rolloff of Audio 2")
        plt.show()
        print("Spectral Rolloff of Audio 1 : ", spectral_rolloff1)
        print("Spectral Rolloff of Audio 2 : ", spectral_rolloff2)

        # MFCC — Mel-Frequency Cepstral Coefficients
        # small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope

        '''
        mfccs1 = librosa.feature.mfcc(x1, sr=sr1)
        mfccs2 = librosa.feature.mfcc(x2, sr=sr2)

        print("MFCC of Audio 1 : ",mfccs1)
        print("MFCC of Audio 2 : ",mfccs2)
        # Displaying  the MFCCs:
        librosa.display.specshow(mfccs1, sr=sr1, x_axis='time')
        plt.show()
        librosa.display.specshow(mfccs2, sr=sr2, x_axis='time')
        plt.show()
        plt.plot(mfccs1[0], label="mfcc1")
        plt.plot(mfccs2[0], label="mfcc2")
        plt.title('Our two MFCC sequences')
        plt.legend()
        plt.show()'''

        mfccs1 = librosa.feature.mfcc(x1, sr=sr1)
        mfccs2 = librosa.feature.mfcc(x2, sr=sr2)
        plt.plot(mfccs1[0], label="MFCC of Original Speech")
        plt.plot(mfccs2[0], label="MFCC of Input Speech")
        plt.title('The two MFCC sequences')
        plt.xlabel("Time")
        plt.ylabel("Mel Frequency Cepstral Coefficients (MFCC)")
        plt.legend()
        plt.show()


        '''
        #dist, cost, path = dtw(mfccs1.T, mfccs2.T)
        l2_norm = lambda mfccs1, mfccs2: (mfccs1 - mfccs2) ** 2
        dist, cost_matrix, acc_cost_matrix, path = dtw(mfccs1.T, mfccs2.T, dist=10)
        print("The normalized distance between the two : ", dist)  # 0 for similar audios'''

        '''
        def _traceback(D):
            i, j = array(D.shape) - 2
            p, q = [i], [j]
            while ((i > 0) or (j > 0)):
                tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
                if (tb == 0):
                    i -= 1
                    j -= 1
                elif (tb == 1):
                    i -= 1
                else:  # (tb == 2):
                    j -= 1
                p.insert(0, i)
                q.insert(0, j)
            return array(p), array(q)

        dist_fun = manhattan_distances
        mfccs1=mfccs1.T
        mfccs2=mfccs2.T
        x=list()
        y=list()
        for line in mfccs1:
            line = list(line)
            x.append(line)

        for line in mfccs2:
            line = list(line)
            y.append(line)

        """
            Computes Dynamic Time Warping (DTW) of two sequences.
            :param array x: N1*M array
            :param array y: N2*M array
            :param func dist: distance used as cost measure
            Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
            """

        assert len(x)  # Report error while x is none
        assert len(y)
        r, c = len(x), len(y)
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
        D1 = D0[1:, 1:]  # view

        for i in range(r):
            for j in range(c):
                D1[i, j] = dist_fun(x[i], y[j])
        C = D1.copy()

        for i in range(r):
            for j in range(c):
                D1[i, j] += min(D0[i, j], D0[i, j + 1], D0[i + 1, j])
        if len(x) == 1:
            path = zeros(len(y)), range(len(y))
        elif len(y) == 1:
            path = range(len(x)), zeros(len(x))
        else:
            path = _traceback(D0)
        #return D1[-1, -1] / sum(D1.shape), C, D1, path
        dist=D1[-1, -1] / sum(D1.shape)
        cost_matrix=C
        acc_cost_matrix=D1
        '''


except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print("Could not request results; {0}".format(e))