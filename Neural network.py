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


#adding water pronunciation spectrograms from various speakers
Sxx_array = np.empty((110,12,450))

for i in range (1,111):
    filename = "data/pronunciation_en_water({0}).wav".format(i)
    audio = wavfile.read(filename)  
    Sxx_padded = createPaddedSpectrogram(audio)
    Sxx_array[i-1] = Sxx_padded


#adding personal spectrogram to the array
#these are pre-recorded on Pre-record
for i in range(0,40):
    audio_data = wavfile.read('data/personal_{0}.wav'.format(i))
    Sxx_padded = createPaddedSpectrogram(audio_data)
    Sxx_array = np.append(Sxx_array, Sxx_padded)


Sxx_array = Sxx_array.reshape(150, 12, 450)
Y_array = np.zeros(150)
Y_array[110:] = 1

xtrain,xtest,ytrain,ytest = model_selection.train_test_split(Sxx_array,Y_array, train_size= 0.9, random_state = 7)

model = Sequential()
model.add(GaussianNoise(0.5, input_shape=(12, 450,)))
model.add(Dense(256))
model.add(Activation('softmax'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('softmax'))
model.add(Dense(128))
model.add(Dense(64))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='nadam')
model.fit(xtrain, ytrain, batch_size=32, epochs=20, validation_data=(xtest,ytest))


#now trying recording test audio
test_data = sayKeyword('water', 'data/testAudio.wav')
test_padded = createPaddedSpectrogram(test_data)


model.predict(test_padded.reshape(1,12,450))
