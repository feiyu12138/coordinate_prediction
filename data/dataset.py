import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from torchvision.transforms import transforms
from data.constant import PHASES
import os
import json
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, args, data_list):
        self.data_list = data_list
        self.transform = transforms.Compose([
            transforms.ColorJitter(),
            transforms.GaussianBlur(kernel_size=5),
            # transforms.RandomRotation(degrees=10),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize((args.image_size,args.image_size)),
            transforms.ToTensor()
        ])
        self.image_size = args.image_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]['image_path']
        img = Image.open(img_path)
        img = self.transform(img)
        target = self.data_list[idx]['target']
        left_centroid = target['left_centroid']
        right_centroid = target['right_centroid']
        target = torch.Tensor(left_centroid + right_centroid)
        return img, target

def read_img_with_annotations(root_img,root_ann):
    data_list = []
    for phase in tqdm(PHASES):
        cur_ann_dir = os.path.join(root_ann, phase)
        cur_img_dir = os.path.join(root_img, phase)
        annotation_dir = os.path.join(cur_ann_dir, "ann")
        img_dir = os.path.join(cur_img_dir, "img")
        for ann_file in tqdm(os.listdir(annotation_dir)):
            ann_path = os.path.join(annotation_dir, ann_file)
            ann = get_two_centroid_point_label(ann_path)
            img_file = ann_file.replace('.json', '')
            img_path = os.path.join(img_dir, img_file)
            data_list.append({'image_path': img_path, 'target': ann})
    return data_list

def get_two_centroid_point_label(json_path):
    label = {}
    with open(json_path, 'r') as f:
        data = json.load(f)
        anns = data['objects']
        for ann in anns:
            if ann['classTitle'] == 'left centroid':
                label['left_centroid'] = ann['points']['exterior'] # [[x, y]]
            elif ann['classTitle'] == 'right centroid':
                label['right_centroid'] = ann['points']['exterior']
            else:
                raise ValueError('Invalid classTitle')
    return label

def read_img_with_annotations_ready(root):
    with open(os.path.join(root, 'info.json'), 'r') as f:
        data_list = json.load(f)
    return data_list

def create_dataset(args,train_ratio=0.8):
    data_list = read_img_with_annotations_ready(args.root)
    train_list = data_list[:int(len(data_list)*train_ratio)]
    val_list = data_list[int(len(data_list)*train_ratio):]
    train_set = CustomDataset(args,train_list)
    val_set = CustomDataset(args,val_list)
    return train_set, val_set

str2list = lambda x: list(map(int, x.split(',')))        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--root_img', type=str, default='/ccvl/net/ccvl15/luoxin/coord/0709')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--n_locations', type=int, default=2)
    parser.add_argument('--root_ann', type=str, default='/ccvl/net/ccvl15/luoxin/coord/0709/ann')
    args = parser.parse_args()
    # train_set, val_set = create_dataset(args)
    data_list = read_img_with_annotations(args.root_img,args.root_ann)
    from ipdb import set_trace; set_trace()

    
    
    
