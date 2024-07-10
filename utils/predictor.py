from typing import Any
from matplotlib import pyplot as plt
import os
class Plotter:
    def __init__(self, output_dir: str, root_dir: str):
        self.output_dir = output_dir
        self.prefix_len = len(root_dir.split('/'))
    def __call__(self, points, images_path):
        points = points.cpu().detach().numpy()
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
        
