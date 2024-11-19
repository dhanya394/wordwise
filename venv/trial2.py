import speech_recognition as sr
import matplotlib.pyplot as plt
import librosa.display
import librosa
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from numpy import array, zeros, argmin, inf, equal, ndim
from os import environ, path
from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *

list = ["yes", "no", "up", "down", "bed", "cat", "dog", "seven", "eight", "nine"]
phon = [['Y', 'EH', 'S'], ['N', 'OW'], ['AH', 'P'], ['D', 'AW', 'N'], ['B', 'EH', 'D'], ['K', 'AE', 'T'], ['D', 'AO', 'G'], ['S', 'EH', 'V', 'AH', 'N'], ['EY', 'T'], ['N', 'AY', 'N']]
wrong_phonemes = []
points = 0

def phoneme_detector(filename):
    MODELDIR = "C:\\Users\\ASUS\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\pocketsphinx\\model"
    DATADIR = "C:\\Users\\ASUS\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\pocketsphinx\\data"

    # Create a decoder with certain model
    config = Decoder.default_config()
    config.set_string('-hmm', path.join(MODELDIR, 'en-us'))
    config.set_string('-allphone', path.join(MODELDIR, 'en-us-phone.lm.bin'))
    config.set_float('-lw', 2.0)
    config.set_float('-beam', 1e-10)
    config.set_float('-pbeam', 1e-10)

    # Decode streaming data.
    decoder = Decoder(config)

    decoder.start_utt()
    stream = open(filename, 'rb')
    while True:
        buf = stream.read(1024)
        if buf:
            decoder.process_raw(buf, False, False)
        else:
            break
    decoder.end_utt()

    hypothesis = decoder.hyp()
    phlist = [seg.word for seg in decoder.seg()]
    newlist = []
    for j in phlist:
        if j != 'SIL':
            newlist.append(j)
    return newlist



for i in range(3):
    word1 = list[i]
    phoneme1 = phon[i]
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say",word1)
        audio = r.listen(source)
    try:
        #word2 = r.recognize_google(audio)
        word2 = r.recognize_sphinx(audio)
        if (word2 == word1):
            print("Congratulations!", word1 ,"is the right Answer! 10 points!")
            points+=10
        else:
            with open("microphone-results.wav", "wb") as f:
                f.write(audio.get_wav_data())
            phoneme2 = phoneme_detector("microphone-results.wav")
            print(phoneme2)
            for k in phoneme1:
                flag = 0
                for l in phoneme2:
                    if(l==k):
                        flag=1
                        break
                if(flag==0):
                    wrong_phonemes.append(k)

    except sr.UnknownValueError:
        #print("Could not understand audio")
        with open("microphone-results.wav", "wb") as f:
            f.write(audio.get_wav_data())
        phoneme2 = phoneme_detector("microphone-results.wav")
        for k in phoneme1:
            flag = 0
            for l in phoneme2:
                if (l == k):
                    flag = 1
                    break
            if (flag==0):
                wrong_phonemes.append(k)
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

print("Phonemes wrongly pronounced : ", wrong_phonemes)
print("Total Points : ",points)