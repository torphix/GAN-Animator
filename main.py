import sys
import yaml
import argparse
import subprocess
from data.face_alignment.face_alignment import Aligner
from data.processor import DataProcessor

from train import train


def update_config_with_args(config_path, args):
    with open(config_path, 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    for key, value in vars(args).items():
        if value is None: continue
        try:
            config[key] = value 
        except:
            raise Exception(f'Arg value:{key} not found in config') 
    return config


if __name__ == '__main__':
    command = sys.argv[1]    
    parser = argparse.ArgumentParser()

    # Dataset Commands
    if command == 'align_dataset':
        '''
        Searchs through directory passed or for videos in 
        datasets directory, aligning any videos it finds
        '''
        parser.add_argument('-ds', '--dataset_path', default='data/datasets',
                            help='''
                            Recusivly finds all videos in directory,
                            looks in datasets if nothing passed''')
        args, leftover_args = parser.parse_known_args()
        aligner = Aligner(args.dataset_path)
        aligner.align_dataset()
        
    if command == 'process_dataset':
        parser.add_argument('-ds', '--dataset_path', default='data/datasets',
                            help='Path to dataset')
        args, leftover_args = parser.parse_known_args()  
        processor = DataProcessor(args)
        processor.process()
        
    elif command == 'train':
        train()
        
    elif command == 'inference':
        pass