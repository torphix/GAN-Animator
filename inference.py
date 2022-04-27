import skvideo.io
from data.utils import process_audio_for_generator

def inference(generator, identity_frame, audio_path, output_path):
    fps = 30
    writer = skvideo.io.FFmpegWriter(output_path,
                                     input_dict={'-r': str(fps)})
    # audio = 
    generator(identity_frame, )
    
    
    
    
def streaming_inference():
    '''Writes to buffer allowing user to read at the same time'''
    
    
    
'''
1. How far along is development
2. Future plans / scope
3. How many features left to implement
4. Frameworks / languages

'''