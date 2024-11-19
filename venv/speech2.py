from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *
import speech_recognition as sr

from os import environ, path
import pyaudio
import wave
import audioop
from collections import deque
import time
import math


def save_speech(self, data, p):
    """
    Saves mic data to temporary WAV file. Returns filename of saved
    file
    """
    filename = 'output_' + str(int(time.time()))
    # writes data to WAV file
    data = b''.join(data)
    wf = wave.open(filename + '.wav', 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)  # TODO make this value a function parameter?
    wf.writeframes(data)
    wf.close()
    return filename + '.wav'

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Speak:")
    audio = r.listen(source)
    print(type(audio))

try:
    #word = r.recognize_google(audio)
    word = r.recognize_sphinx(audio)
    if (word == "seven"):
        print("Right Answer! 10 points!")
    else:
        with open("microphone-results.wav", "wb") as f:
            f.write(audio.get_wav_data())

        MODELDIR = r"C:\Users\ASUS\AppData\Local\Programs\Python\Python36\Lib\site-packages\pocketsphinx\model"
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
        stream = open(r'microphone-results.wav', 'rb')
        while True:
            buf = stream.read(1024)
            if buf:
                decoder.process_raw(buf, False, False)
            else:
                break
        decoder.end_utt()

        hypothesis = decoder.hyp()
        print('Phonemes: ', [seg.word for seg in decoder.seg()])

except sr.UnknownValueError:
    #print("Could not understand audio")
    with open("microphone-results.wav", "wb") as f:
        f.write(audio.get_wav_data())
    MODELDIR = r"C:\Users\ASUS\AppData\Local\Programs\Python\Python36\Lib\site-packages\pocketsphinx\model"
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
    stream = open(r'microphone-results.wav', 'rb')
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
    for i in phlist:
        if i != 'SIL':
            newlist.append(i)
    print('Phonemes: ', newlist)


except sr.RequestError as e:
    print("Could not request results; {0}".format(e))