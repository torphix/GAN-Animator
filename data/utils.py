import os
import math
import torch
import shutil
from tqdm import tqdm
from scipy.io import wavfile
import torch.nn.functional as F
from moviepy.video.io.VideoFileClip import VideoFileClip

def get_dataset(dataset_name):
    if dataset_name == 'crema':
        root = 'avatar/realistic/data/datasets/'
        os.makedirs(f'{root}/raw', exist_ok=True)
        os.chdir(f'{root}/raw')
        os.system('git lfs clone https://github.com/CheyneyComputerScience/CREMA-D.git ')
        os.makedirs(f'{root}/processed/CREMA-D', exist_ok=True)
        shutil.move(f'{root}/raw/CREMA-D/AudioWAV/', f'{root}/processed/CREMA-D/AudioWAV/')
        shutil.move(f'{root}/raw/CREMA-D/VideoFlash/', f'{root}/processed/CREMA-D/VideoFlash/')
        shutil.rmtree(f'{root}/raw/CREMA-D')
    elif dataset_name == 'insertHere':
        pass
    else:
        raise Exception('Error only datasets crema supported ATM')

# Audio
def process_audio_for_generator(seq, cutting_stride, pad_samples, audio_frame_feature_len):
    '''Splits audio into 1:1 frame audio clip mapping (generator input)'''
    pad_left = torch.zeros(pad_samples // 2, 1)
    pad_right = torch.zeros(pad_samples - pad_samples // 2, 1)

    seq = torch.cat((pad_left, seq), 0)
    seq = torch.cat((seq, pad_right), 0)

    stacked = seq.narrow(0, 0, audio_frame_feature_len).unsqueeze(0)
    iterations = (seq.size()[0] - audio_frame_feature_len) // cutting_stride + 1
    for i in range(1, iterations):
        stacked = torch.cat((stacked, seq.narrow(0, i * cutting_stride, audio_frame_feature_len).unsqueeze(0)))
    return stacked

def split_audio(audio, sample_rate, split_size):
    '''
    split_size in seconds: 
    divides audio into chunks {split_size} seconds long
    (discriminator input) 
    '''
    audio = audio.squeeze(1)
    chunk_length = sample_rate * split_size
    chunks = audio.shape[0] / chunk_length
    pad_to = (math.ceil(chunks) * chunk_length) - audio.shape[0]
    audio = F.pad(audio, (0, int(pad_to)), value=0)
    audio = audio.view(-1, int(chunk_length))
    return audio.unsqueeze(1)


def get_audio_max_n_frames(audio_dir, length, overlap):
    print('Getting max number of frames')
    frames = []
    for f in tqdm(os.listdir(audio_dir)):
        samplerate, data = wavfile.read(f'{audio_dir}/{f}')
        stacked = process_audio_for_generator(torch.tensor(data), length, overlap, samplerate)
        frames.append(stacked.shape[0])
    return max(frames)

def get_frame_rate(metadata):
    numerator, denominator  = metadata['video']['@avg_frame_rate'].split('/')
    return int(numerator / denominator)

def shuffle_audio(audio_blocks):
    shuffle_idx = torch.randperm(audio_blocks.size(1))
    audio_blocks = audio_blocks[:, shuffle_idx, :, :]
    return audio_blocks

# Video
def read_video(vid_path):
    vid_data = VideoFileClip(vid_path)
    frames = torch.stack([torch.tensor(frame) for frame in vid_data.iter_frames()])
    return frames.permute(0, 3, 1, 2).contiguous()

def cut_video_sequence(video, frames_per_block):
    # Split video into list of blocks of 5 frames
    # Make video even length
    def _round(x, base=5):
        return base * round(x/base)
    
    crop_to = _round(video.shape[0], base=frames_per_block)
    if crop_to > video.shape[0]: crop_to -= frames_per_block
    video = video[:crop_to, :,:,:]
    video_blocks = torch.split(video, frames_per_block, dim=0)
    return torch.stack(video_blocks, dim=0)

def sample_frames(video, sample_size):
    selection_idx = torch.randperm(sample_size)
    real_frames = video[selection_idx, :, :, :]
    return real_frames


def delete_long_clips(path, max_frames):
    '''
    path: video path eg: data/datasets/processed/lex
    '''
    for folder in os.listdir(f'{path}/VideoFlash'):
        for file in os.listdir(f'{path}/VideoFlash/{folder}/'):
            frames = VideoFileClip(f'{path}/VideoFlash/{folder}/{file}')
            frames = [frame for frame in frames.iter_frames()]
            if len(frames) > max_frames:
                print('removing', file)
                os.remove(f'{path}/VideoFlash/{folder}/{file}')
                os.remove(f'{path}/AudioWAV/{folder}/{file.split(".")[0]}.wav')
            
def crop_to_same(vid_frames, audio_frames):
    if vid_frames.shape[0] > audio_frames.shape[0]:
        vid_frames[:audio_frames.shape[0],:,:,:]
    if audio_frames.shape[0] > vid_frames.shape[0]:
        audio_frames[:vid_frames.shape[0],:,:]
    return vid_frames, audio_frames


    
def ensure_equal(video_path, audio_path):
    audios = [a.split('.')[0] for a in os.listdir(audio_path)]
    videos = [v.split('.')[0] for v in os.listdir(video_path)]
    excess_audios = set(audios) - set(videos)
    excess_videos = set(videos) - set(audios)
    [os.remove(f'{audio_path}/{f}.wav') for f in excess_audios]        
    [os.remove(f'{video_path}/{f}.flv') for f in excess_videos]        
    assert len([a for a in os.listdir(audio_path)]) == len([a for a in os.listdir(video_path)]), \
        'Unequal lengths error in code!!!'
        
        
        