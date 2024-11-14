import sounddevice
import numpy
import threading

MIC_DEVICE = "HDA Intel PCH: ALC255 Analog (hw:1,0)"
MIC_SENSITIVITY = 100
BASE_MOUTH_LENGTH = 30
mouth_length = BASE_MOUTH_LENGTH

def process_sound(indata, _frames, _time, _status):
    global mouth_length
    volume_norm = numpy.linalg.norm(indata)
    mouth_length_bias = int(volume_norm * MIC_SENSITIVITY)
    mouth_length = mouth_length_bias + BASE_MOUTH_LENGTH
    print(mouth_length)

def show_audio_devices():
    devices = sounddevice.query_devices()
    for device in devices:
        if device["max_input_channels"] > 0:
            print(f"Index: {device['index']}, name: \"{device['name']}\"")

def block():
    threading.Event().wait()

def start():
    with sounddevice.InputStream(device=MIC_DEVICE, callback=process_sound, latency=0.1):
        block()
