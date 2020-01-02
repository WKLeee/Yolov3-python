import sys
import os
from timeit import default_timer as timer
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import tensorflow as tf
import cv2


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

FLAGS = {
    'image': False,
    'input': 'V2.MP4',
    'output': ''
}

if __name__ == '__main__':
    image = False
    input_video = 'V2.MP4'
    output = ''
    
    detect_video(YOLO(FLAGS), FLAGS.input, FLAGS.output)

