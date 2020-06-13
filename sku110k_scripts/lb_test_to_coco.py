import argparse
import json
import os
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create dummy json for leaderboard test images')
    parser.add_argument(
        'test_img_dir',
        default="test_images",
        help='Path to the dir where images are stored')
    parser.add_argument(
        '--out-file',
        default="annotations/submission.json",
        help='Output json file')
    args = parser.parse_args()
    return args


def get_filename_as_int(filename):
    try:
        filename = filename.split('_')[1]
        filename = filename.split('.')[0]
        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." %
                         (filename))


def main():
    args = parse_args()

    test = dict()
    test["images"] = []
    test_images = glob(f"{args.test_img_dir}/**.**")
    for file_path in tqdm(test_images):
        img = Image.open(file_path)
        width, height = img.size
        filename = file_path.split("/")[1]
        image_id = get_filename_as_int(filename)
        d = {
            'file_name': filename,
            'height': height,
            'width': width,
            'id': image_id
        }
        test["images"].append(d)
    test["categories"] = [{'supercategory': 'none', 'id': 0, 'name': 'object'}]
    test["type"] = 'type'

    json_fp = open(args.out_file, "w")
    json_str = json.dumps(test)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == '__main__':
    main()
