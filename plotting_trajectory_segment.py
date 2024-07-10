from utils.predictor import Plotter
import argparse
import json

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extractor_time(image_paths):
    time = []
    for image_path in image_paths:
        time.append(image_path.split('/')[-3])
    time = list(set(time))
    return time

def main():
    parser = argparse.ArgumentParser(description='plot trajectory')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='ckpt')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    plotter = Plotter(args.output_dir, args.root) if args.plot else None
    save_dir = args.output_dir + '/trajectory'
    
    points = read_json(args.output_dir+'/coords.json')
    times = extractor_time([point['image_path'] for point in points])
    for time in times:
        point_segment = [point for point in points if time in point['image_path']]
        plotter.set_points_cache(point_segment)
        plotter.plot_trajectory_segment(save_dir)

if __name__ == '__main__':
    main()
    
    