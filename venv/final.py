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


con=sqlite3.Connection('wordwise_db1')
cur=con.cursor()
cur.execute("create table if not exists wordwise(name varchar(20), age varchar(20), gender varchar(20), password varchar(20), username varchar(20) primary key not null)")
print("Table created successfully")

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


window4 = Tk()
window4.title("WordWise")
window4.configure(background="black")


Label(window4, text="WordWise", bg='black', fg='white', font='none 20 bold').pack()
fr5 = Frame()
fr5.config(bg='black')
fr5.pack(side=LEFT, expand=1)

points = 0

def click5():
    def register():
        showinfo('Message', 'Registered successfully')
        name1 = name.get()
        age1 = age.get()
        gender1 = gender.get()
        if gender1==0:
            g = 'Male'
        elif gender1==1:
            g = 'Female'
        else:
            g = 'Other'
        pwd1 = pwd.get()
        username1 = username.get()
        
        #print(name1)
        #print(age1)
        #print(gender1)
        #print(pwd1)
        #print(username1)
        l = (name1, age1, g, pwd1, username1)
        cur.execute("INSERT INTO WORDWISE VALUES (?,?,?,?,?)", l)
        con.commit()

    window5 = Toplevel(window4)
    window5.title("Registration")
    window5.configure(background="black")
    # adding scrollbar
    scrollbar = Scrollbar(window5)
    scrollbar.pack(side=RIGHT, fill=Y)
    Label(window5, text="Registration Details", bg='blue', fg='white', font='none 18 bold').pack()
    fr6 = Frame(window5)
    fr6.config(bg='black')
    fr6.pack(side=LEFT, expand=1)
    Label(fr6, text="Name", bg='black', fg='white', font='none 12 bold').grid(row=0, column=0, padx=10, pady=10,
                                                                              sticky=W)
    name = Entry(fr6, width=20, bg='white')
    name.grid(row=0, column=1, sticky=E)
    Label(fr6, text="Age", bg='black', fg='white', font='none 12 bold').grid(row=1, column=0, padx=10, pady=10,
                                                                             sticky=W)
    age = Entry(fr6, width=20, bg='white')
    age.grid(row=1, column=1, sticky=E)
    Label(fr6, text="Gender", bg='black', fg='white', font='none 12 bold').grid(row=2, column=0, padx=10, pady=10,
                                                                                sticky=W)
    gender = IntVar()
    Radiobutton(fr6, text="Male", variable=gender, value=0, fg="black").grid(row=2, column=1, sticky=E)
    Radiobutton(fr6, text="Female", variable=gender, value=1, fg="black").grid(row=2, column=2, sticky=E)
    Radiobutton(fr6, text="Other", variable=gender, value=2, fg="black").grid(row=2, column=3, sticky=E)
    Label(fr6, text="Password", bg='black', fg='white', font='none 12 bold').grid(row=3, column=0, padx=10, pady=10,
                                                                              sticky=W)
    pwd = Entry(fr6, width=20, bg='white', show='*')
    pwd.grid(row=3, column=1, sticky=E)

    Label(fr6, text="Username", bg='black', fg='white', font='none 12 bold').grid(row=4, column=0, padx=10, pady=10,
                                                                              sticky=W)
    username = Entry(fr6, width=20, bg='white')
    username.grid(row=4, column=1, sticky=E)

    Button(fr6, text='Register', width=20, command=register).grid(row=5, column=1, padx=10, pady=10)
    window5.mainloop()


def click6():
    def login():
        window = Toplevel(window6)
        #window = Tk()
        window.title("WordWise")
        window.configure(background="black")

        Label(window, text="WordWise", bg='black', fg='white', font='none 20 bold').pack()
        fr3 = Frame(window)
        fr3.config(bg='black')
        fr3.pack(side=LEFT, expand=1)

        Label(fr3, text="Points 0", bg='black', fg='blue', font='none 12 bold').grid(row=0, column=2, padx=10, pady=10)

        #points = 0
        block = BooleanVar(window, False)

        def add(x):
            x += 10
            return x

        for i in range(5):
            txt = "Level " + str(i + 1)
            Label(fr3, text=txt, bg='black', fg='blue', font='none 12 bold').grid(row=0, column=1, padx=10, pady=10)
            # audio = sr.AudioData()
            r = sr.Recognizer()
            word1 = list[i]
            phoneme1 = phon[i]

            def click():
                global points
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
                            Label(fr3, text="Points " + str(points), bg='black', fg='blue', font='none 12 bold').grid(
                                row=0,
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

            txt2 = "Say " + word1
            Label(fr3, text=txt2, bg='black', fg='blue', font='none 12 bold').grid(row=1, column=1, padx=10, pady=10)
            a = PhotoImage(file=picpath[i])  # loads image
            Label(fr3, image=a).grid(row=2, column=1, padx=10, pady=10)
            Button(fr3, text='Speak', width=20, command=click).grid(row=3, column=1, padx=10, pady=10)

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
                Label(fr2, text=phn, bg='black', fg='white', font='none 12 bold').grid(row=1, column=0, padx=10,
                                                                                       pady=10,
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

                Label(fr4, text="Choose Word", bg='black', fg='white', font='none 12 bold').grid(row=1, column=0,
                                                                                                 padx=10,
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
                fr8 = Frame(window2)
                fr8.config(bg='black')
                fr8.pack(side=LEFT, expand=1)

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
                    plt.show()
                    plt.plot(ds2)
                    plt.plot(d2)
                    plt.title("VAD2")
                    plt.show()

                Label(fr8, text="Choose Word", bg='black', fg='white', font='none 12 bold').grid(row=1, column=0,
                                                                                                 padx=10,
                                                                                                 pady=10,
                                                                                                 sticky=W)
                variable1 = StringVar(fr8)
                variable1.set("Choose Word")
                word1 = OptionMenu(fr8, variable1, "yes", "no", "up")
                word1.grid(row=2, column=0, sticky=W)
                Button(fr8, text='Detect', width=20, command=detect).grid(row=3, column=0, padx=10, pady=10)
                window2.mainloop()

            Button(fr2, text="Phoneme Detection", width=20, command=click2).grid(row=0, column=0, padx=10, pady=10)
            Button(fr2, text="Voice Activity Detection", width=20, command=click3).grid(row=2, column=0, padx=10,
                                                                                        pady=10)
            Button(fr2, text="MFCC", width=20, command=click4).grid(row=3, column=0, padx=10, pady=10)
            window1.mainloop()

        Label(fr3, text="Good Job! Well Done!", bg='black', fg='blue', font='none 12 bold').grid(row=4, column=1,
                                                                                                 padx=10,
                                                                                                 pady=10)
        Button(fr3, text='View Report', width=20, command=click1).grid(row=5, column=1, padx=10, pady=10)
        window.mainloop()

    window6 = Toplevel(window4)
    window6.title("Login")
    window6.configure(background="black")
    Label(window6, text="Login", bg='blue', fg='white', font='none 18 bold').pack()
    fr7 = Frame(window6)
    fr7.config(bg='black')
    fr7.pack(side=LEFT, expand=1)
    Label(fr7, text="Username", bg='black', fg='white', font='none 12 bold').grid(row=0, column=0, padx=10, pady=10,
                                                                              sticky=W)
    username = Entry(fr7, width=20, bg='white')
    username.grid(row=0, column=1, sticky=E)
    Label(fr7, text="Password", bg='black', fg='white', font='none 12 bold').grid(row=1, column=0, padx=10, pady=10,
                                                                                  sticky=W)
    pwd = Entry(fr7, width=20, bg='white', show='*')
    pwd.grid(row=1, column=1, sticky=E)
    Button(fr7, text='Login', width=20, command=login).grid(row=4, column=1, padx=10, pady=10)
    window6.mainloop()


Button(fr5, text='Register', width=20, command=click5).grid(row=1, column=1, padx=10, pady=10)
Button(fr5, text='Login', width=20, command=click6).grid(row=2, column=1, padx=10, pady=10)

window4.mainloop()

print("Phonemes wrongly pronounced : ", wrong_phonemes)
print("Total Points : ", points)
print("Newlist : ", list)

print("The table values are : ")
cursor = cur.execute("select * from wordwise")
for row in cursor:
    print(str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]))