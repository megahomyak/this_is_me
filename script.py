import sounddevice
import numpy
import threading
import cv2
from mediapipe.python.solutions import pose as mediapipe_pose
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
from pyvirtualcam import Camera

'LEFT_ANKLE', 'LEFT_EAR', 'LEFT_ELBOW', 'LEFT_EYE', 'LEFT_EYE_INNER', 'LEFT_EYE_OUTER', 'LEFT_FOOT_INDEX', 'LEFT_HEEL', 'LEFT_HIP', 'LEFT_INDEX', 'LEFT_KNEE', 'LEFT_PINKY', 'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_WRIST', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'NOSE', 'RIGHT_ANKLE', 'RIGHT_EAR', 'RIGHT_ELBOW', 'RIGHT_EYE', 'RIGHT_EYE_INNER', 'RIGHT_EYE_OUTER', 'RIGHT_FOOT_INDEX', 'RIGHT_HEEL', 'RIGHT_HIP', 'RIGHT_INDEX', 'RIGHT_KNEE', 'RIGHT_PINKY', 'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_WRIST'

MIC_DEVICE = "HDA Intel PCH: ALC255 Analog (hw:1,0)"
MIC_SENSITIVITY = 100
BASE_MOUTH_LENGTH = 30
mouth_length = BASE_MOUTH_LENGTH

def make_color(r, g, b):
    return numpy.array((r, g, b), dtype=numpy.uint8)

FACE_COLOR = make_color(245, 196, 129)
NOSE_COLOR = make_color(198, 158, 104)
MOUTH_COLOR = make_color(0, 0, 0)
EYE_COLOR = make_color(0, 217, 232)

OUTPUT_HEIGHT = 256
OUTPUT_WIDTH = 256

CAMERA_BACKEND = "v4l2loopback"

def make_background():
    return numpy.full((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), numpy.uint8(0))

def process_sound(indata, _frames, _time, _status):
    global mouth_length
    volume_norm = numpy.linalg.norm(indata)
    mouth_length_bias = int(volume_norm * MIC_SENSITIVITY)
    mouth_length = mouth_length_bias + BASE_MOUTH_LENGTH

def middle(mark1, mark2):
    args = {
        attr: (getattr(mark1, attr) + getattr(mark2, attr)) / 2
        for attr in ["x", "y", "z", "visibility"]
    }
    return NormalizedLandmark(**args)

def show_audio_devices():
    devices = sounddevice.query_devices()
    for device in devices:
        if device["max_input_channels"] > 0:
            print(f"Index: {device['index']}, name: \"{device['name']}\"")

def start():
    video_capture = cv2.VideoCapture(0)
    with \
        sounddevice.InputStream(device=MIC_DEVICE, callback=process_sound, latency=0.1), \
        mediapipe_pose.Pose() as pose_recognizer, \
        Camera(width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT, fps=int(video_capture.get(cv2.CAP_PROP_FPS)), backend=CAMERA_BACKEND) as camera:
        while video_capture.isOpened():
            is_success, input_frame = video_capture.read()
            if not is_success:
                break
            output_frame = make_background()

            def draw_rectangle(center, absolute_width, absolute_height, color):
                relative_width = -center.z * absolute_width
                relative_height = -center.z * absolute_height
                print(relative_width, relative_height)
                bottom_row = int((center.y + relative_height/2)*OUTPUT_HEIGHT)
                top_row = int((center.y - relative_height/2)*OUTPUT_HEIGHT)
                left_column = int((center.x - relative_width/2)*OUTPUT_WIDTH)
                right_column = int((center.x + relative_width/2)*OUTPUT_WIDTH)
                output_frame[top_row:bottom_row, left_column:right_column] = color
                print(numpy.count_nonzero(output_frame))

            pose = pose_recognizer.process(input_frame)

            def get_landmark(landmark_name):
                return pose.pose_landmarks.landmark[getattr(mediapipe_pose.PoseLandmark, landmark_name)]

            if pose.pose_landmarks is not None:
                center_of_head = middle(get_landmark("RIGHT_EAR"), get_landmark("LEFT_EAR"))
                center_of_face = get_landmark("NOSE")
                draw_rectangle(center_of_head, 0.3, 0.6, FACE_COLOR)

            camera.send(output_frame)
