from os import environ, path

from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *

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
#stream = open(path.join(DATADIR, 'goforward.raw'), 'rb')
stream = open(r'E:\Major Project stuff\Major Project\Datasets\Speech Commands Dataset\seven\0b40aa8e_nohash_0.wav','rb')
#stream = open(r'E:\Major Project stuff\Major Project\Datasets\Speech Commands Dataset\seven\0c5027de_nohash_0.wav','rb')
#stream = open(r'E:\Major Project stuff\Major Project\Datasets\Speech Commands Dataset\seven\5fe4a278_nohash_0.wav','rb')
while True:
  buf = stream.read(1024)
  if buf:
    decoder.process_raw(buf, False, False)
  else:
    break
decoder.end_utt()

hypothesis = decoder.hyp()
print ('Phonemes: ', [seg.word for seg in decoder.seg()])

import wave
wf = wave.open(r'E:\Major Project stuff\Major Project\Datasets\Speech Commands Dataset\seven\0b40aa8e_nohash_0.wav', 'rb')

print("Channels : ",wf.getnchannels());
print("Sampling rate : ",wf.getframerate());