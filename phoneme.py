from os import environ, path

from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *

MODELDIR = r"C:\Users\ASUS\AppData\Local\Programs\Python\Python36\Lib\site-packages\pocketsphinx\model"
DATADIR = "C:\\Users\\ASUS\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\pocketsphinx\\data"

# Create a decoder with certain model
config = Decoder.default_config()
config.set_string('-hmm', path.join(MODELDIR, 'en-us'))
config.set_string('-allphone', path.join(MODELDIR, 'en-us-phone.lm.bin'))
config.set_string('-lm', path.join(MODELDIR, 'en-us.lm.bin'))
config.set_string('-dict', path.join(MODELDIR, 'cmudict-en-us.dict'))
config.set_float('-lw', 2.0)
config.set_float('-beam', 1e-10)
config.set_float('-pbeam', 1e-10)

# Decode streaming data.
decoder = Decoder(config)

decoder.start_utt()
#stream = open(path.join(DATADIR, 'goforward.raw'), 'rb')
stream = open("E:\\Major Project stuff\\Major Project\\Datasets\\Speech Commands Dataset\\yes\\0ab3b47d_nohash_0.wav",'rb')
#stream = open("E:\\Major Project stuff\\Major Project\\Datasets\\Speech Commands Dataset\\no\\1a9afd33_nohash_0.wav",'rb')
#stream = open("E:\\Major Project stuff\\Major Project\\Datasets\\Speech Commands Dataset\\up\\1e4064b8_nohash_0.wav",'rb')
#stream = open(r'E:\Major Project stuff\Major Project\Datasets\Speech Commands Dataset\seven\0b40aa8e_nohash_0.wav','rb')

while True:
  buf = stream.read(1024)
  if buf:
    decoder.process_raw(buf, False, False)
  else:
    break
decoder.end_utt()

hypothesis = decoder.hyp()
phlist=[seg.word for seg in decoder.seg()]
newlist=[]
for i in phlist:
  if i!='SIL':
    newlist.append(i)
print ('Phonemes: ', newlist)

import wave
wf = wave.open(r'E:\Major Project stuff\Major Project\Datasets\Speech Commands Dataset\seven\0b40aa8e_nohash_0.wav', 'rb')

print("Channels : ",wf.getnchannels());
print("Sampling rate : ",wf.getframerate());
