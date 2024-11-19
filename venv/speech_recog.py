import speech_recognition as sr
print(sr.__version__)
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Speak:")
    audio = r.listen(source)

try:
    #print("You said " + r.recognize_google(audio))
    print("You said " + r.recognize_sphinx(audio))
except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print("Could not request results; {0}".format(e))

'''
harvard = sr.AudioFile('harvard.wav')
with harvard as source:
audio = r.record(source)
#src = sr.AudioFile(r'E:\Major Project stuff\Major Project\Datasets\Speech Commands Dataset\seven\0a7c2a8d_nohash_0.wav')
'''
