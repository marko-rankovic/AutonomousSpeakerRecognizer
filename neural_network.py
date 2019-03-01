import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
from matplotlib import pyplot as plt
import speech_recognition as sr
from wit import Wit

from sklearn import preprocessing, model_selection
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GaussianNoise
from keras.optimizers import Adam


"""
Assisting class that processes sound to the
input more adequate for the neural network
"""
class VoiceProcessing(object):

    def createPaddedSpectrogram(audio):
        f, t ,Sxx = spectrogram(audio[1], fs=44100)
        length = Sxx.shape[1] if Sxx.shape[1] < 450 else 450
        f_cut = f[f<2000.0]
        Sxx = Sxx[:12, :]
        Sxx_padded = np.zeros((12,450))
        Sxx_padded[:12, :length] = Sxx[:12,:length]
    #     plt.figure(figsize=(18,5))
    #     plt.pcolormesh(t,f_cut, Sxx)
    #     plt.show()
        return Sxx_padded


    def sayKeyword(keyword, output_file):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        wit_api = Wit('VE3COSMD3FBL4DBGPHNZJICM7ZCUJ7J5')
        with mic as source:
            print("Adjusting mic...")
            recognizer.adjust_for_ambient_noise(mic, 3)
            print("Say water")
            audio = recognizer.listen(mic, 2, 2.5)
            print("Detecting what you said...")
            response = wit_api.speech(audio.get_wav_data(), None, {'Content-Type': 'audio/wav'})
            if response['_text'] != keyword:
                print('Please speak more clearly.')
            else:
                file = open(output_file,'wb')
                file.write(audio.get_wav_data())
                file.close()
                print("All good!")
        
        audio_data = wavfile.read(output_file)
        return audio_data



"""
The actual implementation of BiometricRecognizer
using Sequential model in Keras library
"""
class SpeakerRecognizer(object):

    def __init__(self, n_samples = 110):
        if n_samples > 110:
            raise ValueError("n_samples should not be bigger than 110")
        
        self.Sxx_array = np.empty((n_samples,12,450))
        self.n_samples = n_samples
        self.n_user_inputs = 0
        self.Y_array = None
        self.model = None

    def prepareNeuralNetworkInput(self):
        """adding spectrograms from various speakers"""
        for i in range (1, self.n_samples):
            filename = "data/pronunciation_en_water({0}).wav".format(i)
            audio = wavfile.read(filename)  
            Sxx_padded = VoiceProcessing.createPaddedSpectrogram(audio)
            self.Sxx_array[i-1] = Sxx_padded
        return self

    def inputUserVoice(self, n_times = 40):
        """
        adding personal spectrogram to the array
        these are pre-recorded on Pre-record
        """
        if n_times > 40:
            raise ValueError("n_times should not be bigger than 40")
    
        self.n_user_inputs = n_times
        for i in range(0, n_times):
            audio_data = wavfile.read('data/personal_{0}.wav'.format(i))
            Sxx_padded = VoiceProcessing.createPaddedSpectrogram(audio_data)
            self.Sxx_array = np.append(self.Sxx_array, Sxx_padded)

        return self


    def trainNetwork(self):
        """Perform model training"""
        self.Sxx_array = self.Sxx_array.reshape(self.n_samples + self.n_user_inputs, 12, 450)
        self.Y_array = np.zeros(self.n_samples + self.n_user_inputs)
        self.Y_array[self.n_samples:] = 1
    
        xtrain,xtest,ytrain,ytest = model_selection.train_test_split(self.Sxx_array, self.Y_array, train_size= 0.9, random_state = 7)

        self.model = Sequential()
        self.model.add(GaussianNoise(0.5, input_shape=(12, 450,)))
        self.model.add(Dense(256))
        self.model.add(Activation('softmax'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256))
        self.model.add(Activation('softmax'))
        self.model.add(Dense(128))
        self.model.add(Dense(64))
        self.model.add(Flatten())
        self.model.add(Dense(1))

        self.model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='nadam')
        self.model.fit(xtrain, ytrain, batch_size=32, epochs=20, validation_data=(xtest,ytest))

        return self

    def testWithVoice(self):
        test_data = VoiceProcessing.sayKeyword('water', 'data/testAudio.wav')
        test_padded = VoiceProcessing.createPaddedSpectrogram(test_data)
        result = self.model.predict(test_padded.reshape(1,12,450))
        print (result)
