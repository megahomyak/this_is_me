import sounddevice
import numpy
import threading
import cv2
from mediapipe.python.solutions import pose as mediapipe_pose
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
from pyvirtualcam import Camera
from copy import copy
from dataclasses import dataclass
import random

'LEFT_ANKLE', 'LEFT_EAR', 'LEFT_ELBOW', 'LEFT_EYE', 'LEFT_EYE_INNER', 'LEFT_EYE_OUTER', 'LEFT_FOOT_INDEX', 'LEFT_HEEL', 'LEFT_HIP', 'LEFT_INDEX', 'LEFT_KNEE', 'LEFT_PINKY', 'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_WRIST', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'NOSE', 'RIGHT_ANKLE', 'RIGHT_EAR', 'RIGHT_ELBOW', 'RIGHT_EYE', 'RIGHT_EYE_INNER', 'RIGHT_EYE_OUTER', 'RIGHT_FOOT_INDEX', 'RIGHT_HEEL', 'RIGHT_HIP', 'RIGHT_INDEX', 'RIGHT_KNEE', 'RIGHT_PINKY', 'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_WRIST'

PIXEL_SIZE = 0.07

MIC_DEVICE = "HDA Intel PCH: ALC255 Analog (hw:1,0)"
MIC_SENSITIVITY = 500
AUDIO_THRESHOLD = 0.5
BASE_MOUTH_LENGTH = 1 * PIXEL_SIZE
mic_volume = 0

def make_color(r, g, b):
    return numpy.array((r, g, b), dtype=numpy.uint8)

FACE_COLOR = make_color(245, 196, 129)
NOSE_COLOR = make_color(198, 158, 104)
MOUTH_COLOR = make_color(0, 0, 0)
EYE_COLOR = make_color(0, 217, 232)
CLOSED_EYE_COLOR = make_color(0, 0, 0)
HAIR_COLOR = make_color(109, 71, 0)
SWEATSHIRT_BASE_COLOR = make_color(0, 170, 0)
SWEATSHIRT_STRIPE_COLOR = make_color(255, 255, 255)
HAND_COLOR = make_color(225, 176, 109)

MIN_BLINKING_RATE = 20
MAX_BLINKING_RATE = 30
blink_stage = 0

SHOW_AUDIO_DEVICES = False

OUTPUT_HEIGHT = 256
OUTPUT_WIDTH = 256

INPUT_TO_OUTPUT = True
INPUT_TO_OUTPUT_FLASHING = False

CAMERA_BACKEND = "v4l2loopback"

def make_background():
    return numpy.full((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), numpy.uint8(0))

def process_sound(indata, _frames, _time, _status):
    global mic_volume
    average_sound = numpy.mean(indata, axis=1)
    mic_volume = numpy.sqrt(numpy.mean(average_sound**2)) * MIC_SENSITIVITY
    if mic_volume < AUDIO_THRESHOLD:
        mic_volume = 0

@dataclass
class Point:
    x: float
    y: float

def average(*points):
    avg = lambda attribute: sum(map(lambda point: getattr(point, attribute), points)) / len(points)
    return Point(x=avg("x"), y=avg("y"))

if SHOW_AUDIO_DEVICES:
    devices = sounddevice.query_devices()
    for device in devices:
        if device["max_input_channels"] > 0:
            print(f"Index: {device['index']}, name: \"{device['name']}\"")
    exit()

def clamp(bound1, value, bound2):
    return min(max(bound1, value), bound2)

def start():
    video_capture = cv2.VideoCapture(0)
    input_to_output_flash_stage = 0
    input_to_output_flash_shown = True
    current_pose = None
    with \
        sounddevice.InputStream(device=MIC_DEVICE, callback=process_sound, latency="low"), \
        mediapipe_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.5, model_complexity=0) as pose_recognizer, \
        Camera(width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT, fps=int(video_capture.get(cv2.CAP_PROP_FPS)) * 2, backend=CAMERA_BACKEND) as camera:
        while video_capture.isOpened():
            is_success, input_frame = video_capture.read()
            if not is_success:
                break
            if INPUT_TO_OUTPUT:
                output_frame = cv2.resize(input_frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
                if INPUT_TO_OUTPUT_FLASHING:
                    input_to_output_flash_stage += 1
                    if input_to_output_flash_stage == 30:
                        input_to_output_flash_shown = not input_to_output_flash_shown
                        input_to_output_flash_stage = 0
            else:
                output_frame = make_background()

            def draw_rectangle_low_level(beginning, end, color):
                if not input_to_output_flash_shown:
                    return
                bottom_row = clamp(0, int(end.y*OUTPUT_HEIGHT), OUTPUT_HEIGHT)
                top_row = clamp(0, int(beginning.y*OUTPUT_HEIGHT), OUTPUT_HEIGHT)
                left_column = clamp(0, int(beginning.x*OUTPUT_WIDTH), OUTPUT_WIDTH)
                right_column = clamp(0, int(end.x*OUTPUT_WIDTH), OUTPUT_WIDTH)
                output_frame[top_row:bottom_row, left_column:right_column] = color

            def draw_rectangle(center, width, height, color):
                width = width * PIXEL_SIZE
                height = height * PIXEL_SIZE
                beginning = Point(center.x - width/2, center.y - height/2)
                end = Point(center.x + width/2, center.y + height/2)
                draw_rectangle_low_level(beginning, end, color)

            def draw_chain(beginning, end, width, height, colors_list):
                for i in range(len(colors_list)):
                    color = colors_list[i]
                    progression = i/(len(colors_list)-1)
                    midpoint = Point(
                        x=(1-progression)*beginning.x + progression*end.x,
                        y=(1-progression)*beginning.y + progression*end.y,
                    )
                    draw_rectangle(midpoint, width, height, color)

            new_pose = pose_recognizer.process(input_frame)

            if new_pose.pose_landmarks is not None:
                current_pose = new_pose

            def get_landmark(landmark_name):
                landmark = current_pose.pose_landmarks.landmark[getattr(mediapipe_pose.PoseLandmark, landmark_name)]
                return Point(landmark.x, landmark.y)

            if current_pose is not None:
                head_center = average(get_landmark("RIGHT_EAR"), get_landmark("LEFT_EAR"))
                nose_center = get_landmark("NOSE")
                face_center = average(head_center, nose_center)
                # Neck
                midshoulders = average(get_landmark("LEFT_SHOULDER"), get_landmark("RIGHT_SHOULDER"))
                midmouth = average(get_landmark("MOUTH_RIGHT"), get_landmark("MOUTH_LEFT"))
                draw_chain(midmouth, midshoulders, 3, 1, [FACE_COLOR]*10)
                # Body
                midhips = average(get_landmark("LEFT_HIP"), get_landmark("RIGHT_HIP"))
                body_width = abs(get_landmark("LEFT_SHOULDER").x - get_landmark("RIGHT_SHOULDER").x)
                draw_chain(midshoulders, midhips, body_width/PIXEL_SIZE, 2, [*[SWEATSHIRT_BASE_COLOR, SWEATSHIRT_STRIPE_COLOR]*4, SWEATSHIRT_BASE_COLOR])
                # Face
                draw_rectangle(head_center, 5, 7, FACE_COLOR)
                # Hair
                hair_top_left = copy(head_center)
                hair_top_left.y -= 3.5 * PIXEL_SIZE
                hair_top_left.x -= 2.5 * PIXEL_SIZE
                hair_bottom_right = copy(hair_top_left)
                hair_bottom_right.x += 5 * PIXEL_SIZE
                hair_bottom_right.y = face_center.y - 1.5 * PIXEL_SIZE
                draw_rectangle_low_level(hair_top_left, hair_bottom_right, HAIR_COLOR)
                # Hair cut
                hair_cut_top_left = copy(hair_top_left)
                hair_cut_top_left.x = clamp(hair_top_left.x, face_center.x + 0.5 * PIXEL_SIZE, hair_bottom_right.x)
                hair_cut_top_left.y = hair_bottom_right.y - 1*PIXEL_SIZE
                hair_cut_bottom_right = copy(hair_cut_top_left)
                hair_cut_bottom_right.x += 1 * PIXEL_SIZE
                hair_cut_bottom_right.y = hair_bottom_right.y
                draw_rectangle_low_level(hair_cut_top_left, hair_cut_bottom_right, FACE_COLOR)
                # Eyes
                for eye_shift in (PIXEL_SIZE, -PIXEL_SIZE):
                    eye_center = copy(face_center)
                    eye_center.x += eye_shift
                    global blink_stage
                    if blink_stage == 0:
                        eye_center.y -= PIXEL_SIZE * 0.8
                        draw_rectangle(eye_center, 1, 1/8, CLOSED_EYE_COLOR)
                    else:
                        eye_center.y -= PIXEL_SIZE
                        draw_rectangle(eye_center, 1, 1, EYE_COLOR)
                if blink_stage == 0:
                    blink_stage = random.randint(MIN_BLINKING_RATE, MAX_BLINKING_RATE)
                else:
                    blink_stage -= 1
                # Mouth
                closed_mouth_center = copy(face_center)
                closed_mouth_center.y += PIXEL_SIZE * 1.5
                mouth_length_bias = mic_volume*PIXEL_SIZE
                mouth_length = BASE_MOUTH_LENGTH + mouth_length_bias
                draw_rectangle_low_level(Point(closed_mouth_center.x - 1.5*PIXEL_SIZE, closed_mouth_center.y), Point(closed_mouth_center.x + 1.5*PIXEL_SIZE, closed_mouth_center.y + mouth_length), MOUTH_COLOR)
                # Nose
                draw_rectangle(nose_center, 1, 1, NOSE_COLOR)
                # Arms
                for beginning, end in (("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"), ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST")):
                    beginning = get_landmark(beginning)
                    end = get_landmark(end)
                    end.y += PIXEL_SIZE
                    beginning.y += PIXEL_SIZE
                    draw_chain(beginning, end, 2, 4, [*[SWEATSHIRT_BASE_COLOR, SWEATSHIRT_STRIPE_COLOR]*2, SWEATSHIRT_BASE_COLOR])
                for hand in ("RIGHT", "LEFT"):
                    hand_center = average(*(get_landmark(f"{hand}_{part}") for part in ("WRIST", "PINKY", "INDEX")))
                    draw_rectangle(hand_center, 3, 3, HAND_COLOR)

            camera.send(output_frame)

start()
