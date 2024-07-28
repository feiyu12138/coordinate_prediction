import os
import argparse
import json
import pickle
from tqdm import tqdm
def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def main(args):
    info_datas = read_json(args.info_path)
    instructions = list(set([data["instruction"] for data in info_datas]))
    # group by instruction
    info_data_groups = {}
    for instruction in instructions:
        group = [data for data in info_datas if data["instruction"] == instruction]
        info_data_groups[instruction] = group
    # sort each group by frame_id
    for instruction, group in info_data_groups.items():
        group.sort(key=lambda x: int(x["frame_id"]))
    os.makedirs(args.output_path, exist_ok=True)
    # save to pickle
    for idx, (instruction, group) in tqdm(enumerate(info_data_groups.items())):
        with open(os.path.join(args.output_path, f'episode_{idx}.pkl'), 'wb') as f:
            pickle.dump(group, f)    
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='group info data by instruction')
    parser.add_argument('--info_path', type=str, default='data/ann/info.json')
    parser.add_argument('--output_path', type=str, default='data/ann')
    args = parser.parse_args()
    main(args)
