
import cv2
import imageio

from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(mp4_file_path, gif_file_path, fps=10):
    clip = VideoFileClip(mp4_file_path)
    clip.write_gif(gif_file_path, fps=fps)

convert_mp4_to_gif('プロジェクト.MP4', 'rapid_scan.gif')


from PIL import Image, ImageSequence

def resize_gif(input_path, output_path, resize_to):
    with Image.open(input_path) as img:
        frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
        resized_frames = []

        for frame in frames:
            resized_frame = frame.resize(resize_to, Image.ANTIALIAS)
            resized_frames.append(resized_frame)

        resized_frames[0].save(output_path, save_all=True, append_images=resized_frames[1:], loop=0)

resize_gif('rapid_scan.gif', 'rapid_scan_resized.gif', (400,200))

