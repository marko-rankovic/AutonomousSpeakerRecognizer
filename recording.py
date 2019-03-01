import speech_recognition as sr

class VoiceRecorder(object):
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

    def RecordKeyword(self, n_inputs = 40):
        if n_inputs > 40:
            raise ValueError("n_inputs should not be bigger than 40")
    
        with self.mic as source:
            print("Adjusting mic...")
            self.recognizer.adjust_for_ambient_noise(self.mic, 3)
            for i in range(0, 40):
                print("Say water: {0}".format(i))
                audio = self.recognizer.listen(self.mic, 2, 2.5)
                file = open('data/personal_{0}.wav'.format(i),'wb')
                file.write(audio.get_wav_data())
                file.close()
                print("All good")

