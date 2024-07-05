import os
import numpy as np
# import skvideo.io

# def _make_dir(filename):
#     folder = os.path.dirname(filename)
#     if not os.path.exists(folder):
#         os.makedirs(folder)

# def save_video(filename, video_frames, fps=60, video_format='mp4'):
#     assert fps == int(fps), fps
#     _make_dir(filename)

#     skvideo.io.vwrite(
#         filename,
#         video_frames,
#         inputdict={
#             '-r': str(int(fps)),
#         },
#         outputdict={
#             '-f': video_format,
#             '-pix_fmt': 'yuv420p', # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
#         }
#     )

# def save_videos(filename, *video_frames, axis=1, **kwargs):
#     ## video_frame : [ N x H x W x C ]
#     video_frames = np.concatenate(video_frames, axis=axis)
#     save_video(filename, video_frames, **kwargs)


## --------------------------------------------
## ------- Vis for Paper, June 13 2024 --------
import imageio
from PIL import Image

def resize_image_mp4(image, block_size=16):
    """Resize image to nearest size divisible by block_size."""
    h, w, _ = image.shape
    h_new = (h // block_size) * block_size
    w_new = (w // block_size) * block_size
    image = np.array(Image.fromarray(image).resize((w_new, h_new)))
    return image

def save_images_to_mp4(image_list, output_file, fps=30, st_sec=1, end_sec=1):
    # Create a writer object specifying the output file, fps, and codec
    writer = imageio.get_writer(output_file, fps=fps, codec='libx264', quality=8)

    for i_m, image in enumerate(image_list):
        # Ensure the image is in uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        # Resize image to be divisible by 16
        image = resize_image_mp4(image)
        ## stay longer at the beginning or ending
        if i_m == 0:
            n_repeats = st_sec * fps
        elif i_m == len(image_list) - 1:
            n_repeats = end_sec * fps
        else:
            n_repeats = 1
        for i_r in range(n_repeats):
            # Write each frame to the video
            writer.append_data(image)

    # Close the writer to finalize the video file
    writer.close()

def read_mp4_to_numpy(video_path):
    # Create a reader object for the video
    reader = imageio.get_reader(video_path)
    
    # Initialize a list to hold all frames
    frames = []

    # Read and append each frame to the list
    for frame in reader:
        frames.append(frame)
    
    reader.close()

    # Convert list of frames to a NumPy array
    video_array = np.stack(frames, axis=0)
    print('video_array:', video_array.shape)

    return video_array