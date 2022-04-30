import os
import yaml
import torch
import numpy as np
import face_alignment
from tqdm import tqdm
from skimage import transform 
from moviepy.editor import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


'''
Crops face from images
VideoFlash
    - sample_set 1
    - sample_set 2
Aligned
    - sample_set 1
'''

def is_video(file):
    file = file.lower()
    if file.endswith('mp4'): return True
    elif file.endswith('mov'): return True
    elif file.endswith('wmv'): return True
    elif file.endswith('avi'): return True
    elif file.endswith('avchd'): return True
    elif file.endswith('flv'): return True
    elif file.endswith('webm'): return True
    elif file.endswith('mkv'): return True
    else: return False
    


class Aligner:
    def __init__(self, input_dir):
        self.root = os.path.abspath('.')
        self.input_dir = f'{self.root}/{input_dir}'
        self.data_config = yaml.load(open(f'{self.root}/configs/data.yaml'),
                                  Loader=yaml.FullLoader)
        self.videos = []
        self.img_size = self.data_config['video']['img_size']
        self.face_anchor_points = [33, 36, 39, 42, 45]
        self.mean_face = np.load(f'{self.root}/data/face_alignment/crema_mean_face.npy')
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fa = face_alignment.FaceAlignment(
                        face_alignment.LandmarksType._2D,
                        flip_input=False,
                        device=device)
    
    def find_videos(self, input_dir):
        # Generator object that finds videos
        for entity in os.listdir(input_dir):
            if os.path.isdir(f'{input_dir}/{entity}'):
                self.find_videos(f'{input_dir}/{entity}')
            elif is_video(f'{input_dir}/{entity}'):
                self.videos.append(f'{input_dir}/{entity}')
        return self.videos
    
    def align_dataset(self):
        print('Aligning images...')
        videos = self.find_videos(self.input_dir)
        for video in tqdm(videos):
            clip = VideoFileClip(video)
            frames = torch.stack([torch.tensor(frame) for frame in clip.iter_frames()])
            new_frames = self.align_frames(frames)
            if new_frames is None: continue
            new_clip = ImageSequenceClip(new_frames, fps=clip.fps)
            new_clip.write_videofile(f'{video.split(".")[0]}.mp4', fps=clip.fps)
            os.remove(video)
            
    def align_frames(self, frames):
        new_frames = []
        src = self.fa.get_landmarks_from_image(frames[0])
        if src is None: return
        else: src = src[0][self.face_anchor_points, :]
        dst = self.mean_face[self.face_anchor_points, :]
        for frame in frames:
            transformation = transform.estimate_transform('similarity', src, dst)
            warped = transform.warp(frame, 
                                    inverse_map=transformation.inverse,
                                    output_shape=self.img_size)
            warped *= 255
            new_frames.append(warped)        
        return new_frames