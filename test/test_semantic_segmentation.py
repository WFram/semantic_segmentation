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

    def get_predictions(self, data_sample):
        pixel_data = data_sample.pred_sem_seg
        predictions = pixel_data.values()[0].cpu().numpy()
        return predictions[0]

    def infer(self, image: np.ndarray) -> None:
        segmentation_result = inference_model(self.model, image)

        semantic_image = self.get_predictions(segmentation_result)
        semantic_image = semantic_image.astype(np.uint8)

        print("Success!")

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

    parser.add_argument('--image',
                        dest='image',
                        type=str,
                        required=True,
                        help='Path to test image')

    parser.add_argument('--output_dir',
                        dest='output_dir',
                        type=str,
                        required=False,
                        help='Path to output folder')
    
    options = parser.parse_args()

    semantic_segmentation = SemanticSegmentation(options)
    semantic_segmentation.infer(options.image)
