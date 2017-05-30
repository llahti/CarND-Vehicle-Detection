import cv2
import moviepy.editor as mpy
# from moviepy.editor import VideoFileClip
from CarFinder.carfinder import CarFinder
import utils
import numpy as np


input_file = "project_video.mp4"
output_file = "./output_videos/augmented_project_video2.mp4"

# Open video clip
clip = mpy.VideoFileClip(input_file)

# Get frame iterator which will be used in "frame_generator()"
clip_iterator = clip.iter_frames()

# Create CarFinder Object
cf = CarFinder()


def frame_generator(t):
    """This function generates annotated frames for video"""

    # Get next frame
    frame = next(clip_iterator)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Run frame through lane finder
    cf.find(frame)
    bboxes = cf.draw_bounding_boxes(frame)

    # Draw per frame heatmap
    bboxes = cf.pip_heatmap_per_frame(bboxes, (10, 5), (213, 120))

    # draw averaged heatmap
    bboxes = cf.pip_heatmap_averaged(bboxes, (10, 150), (213, 120))

    # Draw heatmap threshold
    bboxes = cf.pip_heatmap_threshold(bboxes, (250, 5), (213, 120))

    # Draw labels
    bboxes = cf.pip_labels(bboxes, (250, 150), (213, 120))

    # Need to convert from BGR to RGB to get colors right in video
    output_image = cv2.cvtColor(bboxes, cv2.COLOR_BGR2RGB)
    return output_image

# Write video file to disk
augmented_video = mpy.VideoClip(frame_generator, duration=clip.duration)
augmented_video.write_videofile(output_file, fps=clip.fps)