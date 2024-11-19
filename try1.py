import automatic_speech_recognition as asr

file = 'E:\Major Project stuff\Major Project\Datasets\english_children\english_children\english_free_speech\files_cut_by_sentences\01_M_native\a boy looking at the frog.wav'  # sample rate 16 kHz, and 16 bit depth
sample = asr.utils.read_audio(file)
pipeline = asr.load('deepspeech2', lang='en')
pipeline.model.summary()     # TensorFlow model
sentences = pipeline.predict([sample])
print(sentences)