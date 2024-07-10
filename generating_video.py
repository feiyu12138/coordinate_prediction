import argparse
from utils.video import VideoGenerator
from data.dataset import read_img_with_annotations_ready
import json

replace_prefix = lambda x,y,z: z + '/' + '/'.join(x.split('/')[y:])

def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--image_dir', type=str, default='ckpt')
    parser.add_argument('--output_path', type=str, default='ckpt')
    args = parser.parse_args()
    prefix_len = len(args.root.split('/'))
    data_list = read_img_with_annotations_ready(args.root)
    image_paths = [replace_prefix(d['image_path'],prefix_len,args.image_dir) for d in data_list]
    image_paths = [image for image in image_paths if "left" in image]
    video_generator = VideoGenerator(image_paths)
    video_generator.generate_video(args.output_path)
    
if __name__ == '__main__':
    main()