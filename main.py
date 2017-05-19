import cv2
import moviepy.editor as mpy
from moviepy.editor import VideoFileClip
from CarFinder.carfinder import CarFinder
import utils
import numpy as np


input_file = "test_video.mp4"
output_file = "augmented_project_video.mp4"


clip = mpy.VideoFileClip(input_file)

# We need to create iterator from udacam which will be used in below
# function to get frames from camera
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

    # Add per frame heatmap to result image
    colorized_heatmap = cf.heatmap_raw.astype(dtype=np.uint8)*100
    colorized_heatmap = cv2.cvtColor(colorized_heatmap, cv2.COLOR_GRAY2BGR)
    utils.pip(bboxes, colorized_heatmap, (0,0), (213, 120), 5)


    # Need to convert from BGR to RGB to get colors right in video
    output_image = cv2.cvtColor(bboxes, cv2.COLOR_BGR2RGB)
    return output_image

# Write video file to disk
augmented_video = mpy.VideoClip(frame_generator, duration=clip.duration)
augmented_video.write_videofile(output_file, fps=clip.fps)