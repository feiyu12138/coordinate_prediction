from utils.predictor import Plotter
import argparse
import json

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def main():
    parser = argparse.ArgumentParser(description='plot trajectory')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='ckpt')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    
    points = read_json(args.output_dir+'/coords.json')
    plotter = Plotter(args.output_dir, args.root, points) if args.plot else None
    save_dir = args.output_dir + '/trajectory'
    plotter.plot_trajectory(save_dir)

if __name__ == '__main__':
    main()
    
    