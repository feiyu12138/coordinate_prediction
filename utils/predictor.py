from typing import Any
from matplotlib import pyplot as plt
import os
from data.constant import PHASES
import numpy as np

class Plotter:
    def __init__(self, output_dir: str, root_dir: str, point_cache: Any = None):
        self.output_dir = output_dir
        self.prefix_len = len(root_dir.split('/'))
        self.points_cache = point_cache
    def __call__(self, points, images_path):
        points = points.cpu().detach().numpy()
        self.points_cache = points
        if not isinstance(images_path, list):
            images_path = [images_path]
        for i, img_path in enumerate(images_path[0]):
            img_name = img_path.split('/')[self.prefix_len:]
            img_name = '/'.join(img_name)
            img = plt.imread(img_path)
            plt.imshow(img)
            for point in points[i]:
                plt.scatter(point[0], point[1], c='r')
            save_path = f'{self.output_dir}/{img_name}'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        
    def set_points_cache(self, points):
        self.points_cache = points
        
    def plot_trajectory_segment(self, save_path=None):
        sorted_list = []
        sorted_list = [point for point in self.points_cache if 'left' in point['image_path']]
        first_image = plt.imread(sorted_list[0]['image_path'])
        trajectory = np.array([point["coords"] for point in sorted_list])
        plt.imshow(first_image)
        plt.scatter(trajectory[:,:,0], trajectory[:,:,1],c='r')
        if save_path:
            save_name = sorted_list[0]['image_path'].split('/')[self.prefix_len-1:-2]
            save_name = '/'.join(save_name)
            save_p = f'{save_path}/{save_name}.png'
            os.makedirs(os.path.dirname(save_p), exist_ok=True)
            plt.savefig(save_p)
        plt.close()
        return trajectory, first_image
            
    def plot_trajectory(self, save_path=None):
        for phase in PHASES:
            sorted_list = []
            sorted_list.extend(sorted([point for point in self.points_cache if phase in point['image_path']],key=lambda x: x['image_path']))
            sorted_list = [point for point in sorted_list if 'left' in point['image_path']]
            first_image = plt.imread(sorted_list[0]['image_path'])
            trajectory = np.array([point["coords"] for point in sorted_list])
            plt.imshow(first_image)
            plt.plot(trajectory[:,:,0], trajectory[:,:,1], 'r')
            if save_path:
                save_p = f'{save_path}/{phase}.png'
                plt.savefig(save_p)
        return trajectory, first_image
        
class Predictor:
    def __init__(self, model, norm, plotter=None):
        self.model = model
        self.norm = norm
        self.plotter = plotter
    def __call__(self, image, image_path):
        coords, _ = self.model(image)
        coords = self.norm.denormalize(coords)
        if self.plotter:
            self.plotter(coords,image_path)
        return coords
        
