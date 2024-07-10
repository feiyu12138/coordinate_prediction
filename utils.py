import torch
from data.constant import PHASES
import os
from tqdm import tqdm
import json

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

str2list = lambda x: list(map(int, x.split(',')))        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--root_img', type=str, default='/ccvl/net/ccvl15/luoxin/coord/0709')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--n_locations', type=int, default=2)
    parser.add_argument('--root_ann', type=str, default='/ccvl/net/ccvl15/luoxin/coord/ann')
    args = parser.parse_args()
    # train_set, val_set = create_dataset(args)
    data_list = read_img_with_annotations(args.root_img,args.root_ann)
    info_path = os.path.join(args.root_ann, 'info.json')
    with open(info_path, 'w') as f:
        json.dump(data_list, f,indent=4)
    from ipdb import set_trace; set_trace()