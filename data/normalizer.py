import torch
from data.constant import IMAGE_SIZE
class Normalizer:
    def __init__(self, image_size=IMAGE_SIZE, device='cuda'):
        self.image_size = torch.Tensor(image_size).to(device)
        self.device = device

    def normalize(self, tensor):
        if isinstance(tensor, list):
            tensor = torch.Tensor(tensor).to(self.device)
        return (tensor * 2 + 1) / self.image_size - 1
    
    def denormalize(self, tensor):
        if isinstance(tensor, list):
            tensor = torch.Tensor(tensor).to(self.device)
        return ((tensor + 1) * self.image_size - 1) / 2