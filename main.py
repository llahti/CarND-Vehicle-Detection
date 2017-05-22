import cv2
import moviepy.editor as mpy
# from moviepy.editor import VideoFileClip
from CarFinder.carfinder import CarFinder
import utils
import numpy as np


input_file = "project_video.mp4"
output_file = "result_video_heavy_overlap.mp4"

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
    frame_heatmap = cf.heatmap_raw.copy().astype(dtype=np.uint8)
    frame_heatmap = cv2.cvtColor(frame_heatmap, cv2.COLOR_GRAY2BGR) * 15
    bboxes = utils.pip(bboxes, frame_heatmap, (10, 5), (213, 120), 5,
                       "Per Frame Heatmap")

    # draw averaged heatmap
    avg_heat = cf.heatmap_averaged.mean()
    avg_heat = cv2.cvtColor(avg_heat.astype(dtype=np.uint8),
                            cv2.COLOR_GRAY2BGR) * 15
    bboxes = utils.pip(bboxes, avg_heat, (10, 150), (213, 120), 5,
                       "Average Heatmap")

    # Draw heatmap threshold
    heat_thold = cf.heatmap_threshold
    heat_thold = cv2.cvtColor(heat_thold.astype(dtype=np.uint8),
                              cv2.COLOR_GRAY2BGR) * 15
    bboxes = utils.pip(bboxes, heat_thold, (250, 5), (213, 120), 5,
                       "Heatmap Threshold")
    # Draw labels
    labels = cf.labels[0] * 20  # Idiotic way to make labels visible
    labels = cv2.cvtColor(labels.astype(dtype=np.uint8),
                          cv2.COLOR_GRAY2BGR) * 15
    bboxes = utils.pip(bboxes, labels, (250, 150), (213, 120), 5,
                       "Labels")

    # Need to convert from BGR to RGB to get colors right in video
    output_image = cv2.cvtColor(bboxes, cv2.COLOR_BGR2RGB)
    return output_image

# Write video file to disk
augmented_video = mpy.VideoClip(frame_generator, duration=clip.duration)
augmented_video.write_videofile(output_file, fps=clip.fps)