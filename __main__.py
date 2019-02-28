import neural_network as nn
import recording as rec


if __name__ == "__main__":
    recorder = rec.VoiceRecorder()
    recorder.RecordKeyword()
    model = nn.SpeakerRecognizer().prepareNeuralNetworkInput().inputUserVoice().trainNetwork()
    model.testWithVoice()