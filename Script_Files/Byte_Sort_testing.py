import sys
import numpy as np
import matplotlib.pyplot as plt
np.float=float
sys.path.append("D:/Hackathon/tn_hack/ByteTrack")
from dataclasses import dataclass
from typing import Generator
import cv2


SOURCE_VIDEO_PATH = "D:/Hackathon/tn_hack/videoplayback2.mp4"


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch

import torch
WEIGHTS_PATH="D:/Hackathon/tn_hack/weights/yolov5s.pt"
model = torch.hub.load('D:/git_repositories/yolov5', 'custom', WEIGHTS_PATH,source='local')
byte_tracker = BYTETracker(BYTETrackerArgs())


def generate_frames(video_file: str) -> Generator[np.ndarray, None, None]:
    video = cv2.VideoCapture(video_file)

    while video.isOpened():
        success, frame = video.read()

        if not success:
            break

        yield frame

    video.release()
    
def plot_image(image: np.ndarray, size: int = 12) -> None:
    plt.figure(figsize=(size, size))
    plt.imshow(image[...,::-1])
    plt.show()
     
    

     
frame_iterator = iter(generate_frames(video_file=SOURCE_VIDEO_PATH))
frame = next(frame_iterator)
plot_image(frame, 16)


