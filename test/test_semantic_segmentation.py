from mmseg.apis import inference_model, init_model, show_result_pyplot

import sys
import cv2
import re
import numpy as np
import time
import argparse


class SemanticSegmentation(object):
    def __init__(self, options):
        self.config_file = options.config_file
        self.checkpoint_file = options.checkpoint_file

        # TODO: make device adjustable
        self.model = init_model(self.config_file, self.checkpoint_file, device='cuda:0')

        print("NN: Start processing")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config_file',
                        dest='config_file',
                        type=str,
                        required=True,
                        help='Path to config')

    parser.add_argument('--checkpoint_file',
                        dest='checkpoint_file',
                        type=str,
                        required=True,
                        help='Path to weights')
    
    options = parser.parse_args()

    seg = SemanticSegmentation(options)
