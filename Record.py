import speech_recognition as sr

recognizer = sr.Recognizer()
mic = sr.Microphone

for i in range(0, 40):
    with mic as source:
        print("Adjusting mic...")
        recognizer.adjust_for_ambient_noise(mic, 3)
        print("Say water")
        audio = recognizer.listen(mic, 2, 2.5)
        file = open('data/personal_{0}.wav'.format(i),'wb')
        file.write(audio.get_wav_data())
        file.close()
        print("All good")