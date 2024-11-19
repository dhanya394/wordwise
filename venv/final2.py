from tkinter import *
from tkinter.messagebox import *
import sqlite3
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
from VAD import VoiceActivityDetector
import numpy

from keras.models import load_model
model=load_model('E:/Major Project stuff/ASR/model/best_model2.hdf5')

classes = ['bed', 'cat', 'dog', 'down', 'eight', 'nine', 'no', 'seven', 'up', 'yes']

def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=numpy.argmax(prob[0])
    return classes[index]


list = ["yes", "no", "up", "down", "bed", "cat", "dog", "seven", "eight", "nine"]
list1 = ["bird", "five", "four", "go", "happy", "house", "left", "marvin", "off", "on", "one", "right", "sheila", "six",
         "stop", "three", "tree", "two", "wow", "zero"]
phon = [['Y', 'EH', 'S'], ['N', 'OW'], ['AH', 'P'], ['D', 'AW', 'N'], ['B', 'EH', 'D'], ['K', 'AE', 'T'],
        ['D', 'AO', 'G'], ['S', 'EH', 'V', 'AH', 'N'], ['EY', 'T'], ['N', 'AY', 'N']]
phone2 = [['B', 'ER', 'D'], ['F', 'AY', 'V'], ['F', 'AO', 'R'], ['G', 'OW'], ['HH', 'AE', 'P', 'IY'], ['HH', 'AW', 'S'],
          ['L', 'EH', 'F', 'T'], ['M', 'AA', 'R', 'V', 'IH', 'N'],
          ['AO', 'F'], ['AA', 'N'], ['W', 'AH', 'N'], ['R', 'AY', 'T'], ['SH', 'IY', 'L', 'AH'], ['S', 'IH', 'K', 'S'],
          ['S', 'T', 'AA', 'P'], ['TH', 'R', 'IY'],
          ['T', 'R', 'IY'], ['T', 'UW'], ['W', 'AW'], ['Z', 'IY', 'R', 'OW']]
wrong_phonemes = []
picpath = ["E:\\Major Project stuff\\Major Project\\pictures\\yes.gif",
           "E:\\Major Project stuff\\Major Project\\pictures\\no.gif",
           "E:\\Major Project stuff\\Major Project\\pictures\\up.gif",
           "E:\\Major Project stuff\\Major Project\\pictures\\down.gif",
           "E:\\Major Project stuff\\Major Project\\pictures\\bed.gif",
           "E:\\Major Project stuff\\Major Project\\pictures\\cat.gif",
           "E:\\Major Project stuff\\Major Project\\pictures\\dog.gif",
           "E:\\Major Project stuff\\Major Project\\pictures\\seven.gif",
           "E:\\Major Project stuff\\Major Project\\pictures\\eight.gif",
           "E:\\Major Project stuff\\Major Project\\pictures\\nine.gif"]

word_path = {"yes": r"E:\Major Project stuff\Major Project\Datasets\Speech Commands Dataset\yes\0a7c2a8d_nohash_0.wav",
             "no": r"E:\Major Project stuff\Major Project\Datasets\Speech Commands Dataset\no\0ab3b47d_nohash_0.wav",
             "up": r"E:\Major Project stuff\Major Project\Datasets\Speech Commands Dataset\up\0bd689d7_nohash_0.wav"}


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


window = Tk()
window.title("WordWise")
window.configure(background="black")

Label(window, text="WordWise", bg='black', fg='white', font='none 20 bold').pack()
fr1 = Frame()
fr1.config(bg='black')
fr1.pack(side=LEFT, expand=1)

Label(fr1, text="Points 0", bg='black', fg='blue', font='none 12 bold').grid(row=0, column=2, padx=10, pady=10)

points = 0
block = BooleanVar(window, False)


def add(x):
    x += 10
    return x


for i in range(5):
    txt = "Level " + str(i + 1)
    Label(fr1, text=txt, bg='black', fg='blue', font='none 12 bold').grid(row=0, column=1, padx=10, pady=10)
    # audio = sr.AudioData()
    r = sr.Recognizer()
    word1 = list[i]
    phoneme1 = phon[i]


    def click():
        global points
        import sounddevice as sd
        import soundfile as sf

        samplerate = 16000
        duration = 1  # seconds
        filename = 'voice1.wav'
        print("start")
        mydata = sd.rec(int(samplerate * duration), samplerate=16000,
                        channels=1, blocking=True)
        print("end")
        sd.wait()
        sf.write(filename, mydata, samplerate)

        filepath = 'C:/Users/ASUS/PycharmProjects/Speech_Assessment_Tool/venv'
        samples, sample_rate = librosa.load(filepath + '/' + 'voice1.wav', sr=16000)
        samples = librosa.resample(samples, sample_rate, 8000)

        word2 = predict(samples)

        if (word2 == word1):
            points = add(points)
            showinfo('Bravo!', 'Congratulations! Its the right answer!')
            print("Congratulations!", word1, " is the right Answer!", points, "points!")
            Label(fr1, text="Points " + str(points), bg='black', fg='blue', font='none 12 bold').grid(row=0,
                                                                                                      column=2,
                                                                                                      padx=10,
                                                                                                      pady=10)

        else:
            phoneme2 = phoneme_detector("microphone-results.wav")
            for k in phoneme1:
                flag = 0
                for l in phoneme2:
                    if (l == k):
                        flag = 1
                        break
                if (flag == 0):
                    wrong_phonemes.append(k)
        block.set(False)

        '''
        with sr.Microphone() as source:
            audio = r.listen(source)
            with open("microphone-results.wav", "wb") as f:
                f.write(audio.get_wav_data())
            try:
                src = sr.AudioFile("microphone-results.wav")
                with src as source:
                    audio1 = r.record(source)
                # word2 = r.recognize_google(audio1)
                word2 = r.recognize_sphinx(audio1)
                

                if (word2 == word1):
                    points = add(points)
                    showinfo('Bravo!', 'Congratulations! Its the right answer!')
                    print("Congratulations!", word1, " is the right Answer!", points, "points!")
                    Label(fr1, text="Points " + str(points), bg='black', fg='blue', font='none 12 bold').grid(row=0,
                                                                                                              column=2,
                                                                                                              padx=10,
                                                                                                              pady=10)
                else:
                    phoneme2 = phoneme_detector("microphone-results.wav")
                    for k in phoneme1:
                        flag = 0
                        for l in phoneme2:
                            if (l == k):
                                flag = 1
                                break
                        if (flag == 0):
                            wrong_phonemes.append(k)

            except sr.UnknownValueError:
                # print("Could not understand audio")
                phoneme2 = phoneme_detector("microphone-results.wav")
                for k in phoneme1:
                    flag = 0
                    for l in phoneme2:
                        if (l == k):
                            flag = 1
                            break
                    if (flag == 0):
                        wrong_phonemes.append(k)
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
        block.set(False)
        '''


    txt2 = "Say " + word1
    Label(fr1, text=txt2, bg='black', fg='blue', font='none 12 bold').grid(row=1, column=1, padx=10, pady=10)
    a = PhotoImage(file=picpath[i])  # loads image
    Label(fr1, image=a).grid(row=2, column=1, padx=10, pady=10)
    Button(fr1, text='Speak', width=20, command=click).grid(row=3, column=1, padx=10, pady=10)

    # print("Say",word1)

    block.set(True)
    window.wait_variable(block)

phn = "Phonemes Wrongly Pronounced : "
for q in wrong_phonemes:
    phn = phn + q + " "

for p in wrong_phonemes:
    for q in range(20):
        ph = phone2[q]
        for r in ph:
            if (r == p):
                list.append(list1[q])


def click1():
    window1 = Toplevel(window)
    window1.title("Report")
    window1.configure(background="black")
    Label(window1, text="Report", bg='blue', fg='white', font='none 18 bold').pack()
    fr2 = Frame(window1)
    fr2.config(bg='black')
    fr2.pack(side=LEFT, expand=1)

    def click2():
        Label(fr2, text=phn, bg='black', fg='white', font='none 12 bold').grid(row=1, column=0, padx=10, pady=10,
                                                                               sticky=W)

    def click4():
        window3 = Toplevel(window1)
        window3.title("MFCC Sequences")
        window3.configure(background="black")
        Label(window3, text="MFCC Sequences", bg='blue', fg='white', font='none 18 bold').pack()
        fr4 = Frame(window3)
        fr4.config(bg='black')
        fr4.pack(side=LEFT, expand=1)

        def get():
            r = sr.Recognizer()
            with sr.Microphone() as source:
                audio2 = r.listen(source)
                with open("microphone-results2.wav", "wb") as f:
                    f.write(audio2.get_wav_data())
            wrd = variable2.get()
            inp1 = word_path[wrd]
            inp2 = "microphone-results2.wav"
            x1, sr1 = librosa.load(inp1, sr=44100)
            x2, sr2 = librosa.load(inp2, sr=44100)
            mfccs1 = librosa.feature.mfcc(x1, sr=sr1)
            mfccs2 = librosa.feature.mfcc(x2, sr=sr2)
            plt.plot(mfccs1[0], label="mfcc1")
            plt.plot(mfccs2[0], label="mfcc2")
            plt.title('The two MFCC sequences')
            plt.legend()
            plt.show()

        Label(fr4, text="Choose Word", bg='black', fg='white', font='none 12 bold').grid(row=1, column=0, padx=10,
                                                                                         pady=10,
                                                                                         sticky=W)
        variable2 = StringVar(fr4)
        variable2.set("Choose Word")
        word2 = OptionMenu(fr4, variable2, "yes", "no", "up")
        word2.grid(row=2, column=0, sticky=W)
        Button(fr4, text='Speak', width=20, command=get).grid(row=3, column=0, padx=10, pady=10)
        window3.mainloop()

    def click3():
        window2 = Toplevel(window1)
        window2.title("Voice Activity Detection")
        window2.configure(background="black")
        Label(window2, text="Voice Activity Detection", bg='blue', fg='white', font='none 18 bold').pack()
        fr3 = Frame(window2)
        fr3.config(bg='black')
        fr3.pack(side=LEFT, expand=1)

        # block1 = BooleanVar(window2, False)
        def detect():
            r = sr.Recognizer()
            with sr.Microphone() as source:
                audio2 = r.listen(source)
                with open("microphone-results1.wav", "wb") as f:
                    f.write(audio2.get_wav_data())
            wrd = variable1.get()
            inp1 = word_path[wrd]
            inp2 = "microphone-results1.wav"
            v1 = VoiceActivityDetector(inp1)
            v2 = VoiceActivityDetector(inp2)
            raw_detection = v1.detect_speech()
            speech_labels1 = v1.convert_windows_to_readable_labels(raw_detection)
            print("These are the speech labels of normal speech input : ", speech_labels1)
            raw_detection = v2.detect_speech()
            speech_labels2 = v2.convert_windows_to_readable_labels(raw_detection)
            print("These are the speech labels of the child's speech input : ", speech_labels2)
            # block1.set(True)
            # window2.wait_variable(block1)
            ds1, d1 = v1.plot_detected_speech_regions()
            ds2, d2 = v2.plot_detected_speech_regions()
            plt.plot(ds1)
            plt.plot(d1)
            plt.title("VAD1")
            # plt.show()
            plt.plot(ds2)
            plt.plot(d2)
            plt.title("VAD2")
            plt.show()

        Label(fr3, text="Choose Word", bg='black', fg='white', font='none 12 bold').grid(row=1, column=0, padx=10,
                                                                                         pady=10,
                                                                                         sticky=W)
        variable1 = StringVar(fr3)
        variable1.set("Choose Word")
        word1 = OptionMenu(fr3, variable1, "yes", "no", "up")
        word1.grid(row=2, column=0, sticky=W)
        Button(fr3, text='Detect', width=20, command=detect).grid(row=3, column=0, padx=10, pady=10)
        window2.mainloop()

    Button(fr2, text="Phoneme Detection", width=20, command=click2).grid(row=0, column=0, padx=10, pady=10)
    Button(fr2, text="Voice Activity Detection", width=20, command=click3).grid(row=2, column=0, padx=10, pady=10)
    Button(fr2, text="MFCC", width=20, command=click4).grid(row=3, column=0, padx=10, pady=10)
    window1.mainloop()


Label(fr1, text="Good Job! Well Done!", bg='black', fg='blue', font='none 12 bold').grid(row=4, column=1, padx=10,
                                                                                         pady=10)
Button(fr1, text='View Report', width=20, command=click1).grid(row=5, column=1, padx=10, pady=10)
window.mainloop()

print("Phonemes wrongly pronounced : ", wrong_phonemes)
print("Total Points : ", points)
print("Newlist : ", list)