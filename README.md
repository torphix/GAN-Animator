# Speech driven animation (Work in progress)
Based on [this](https://arxiv.org/pdf/1805.09313.pdf) paper.

A generative adversarial network that learns to generate videos of a human speaking
given an input identity frame and audio speech file

Building this as wanted to learn how to train GANS and am using it as part of a larger project

## Data
Supported -> CREMA-D, Custom

To train a dataset of short video clips and corresponding speech audio clips are required.
1. Format Data
Data should be formatted like:
    - data/{NAME_OF_DATASET}/datasets/AudioWAV
    - data/{NAME_OF_DATASET}/datasets/VideoFlash

2. Align data
Once have been formatted correctly run command edit the file configs/data.yaml changing the name parameter to {NAME_OF_DATASET}:
Then run: `python main.py align_dataset` to center align the videos (note the aligner uses the first frame to align the rest of the video this can cause problems if the human moves out of the frame whilst speaking eg: video clips should be cleaned so that the speaker is always roughly in the same spot with minimal movement)

3. Process data
Create the dataset by running: `python main.py process_dataset` once this is done you can move onto training


## Train
Run: `python main.py train` to begin training