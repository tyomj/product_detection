import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

START_BOUNDING_BOX_ID = 1
COLUMNS = ["filename", "x1", "y1", "x2", "y2", "cls", "w", "h"]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert SKU110k dataset to COCO-style')
    parser.add_argument(
        '--in-train-path',
        default="annotations/annotations_train.csv",
        help='Path to the train csv file')
    parser.add_argument(
        '--in-val-path',
        default="annotations/annotations_val.csv",
        help='Path to the val csv file')
    parser.add_argument(
        '--in-test-path',
        default="annotations/annotations_test.csv",
        help='Path to the test csv file')
    parser.add_argument(
        '--out-train-path',
        default="annotations/train.json",
        help='Path to save train json file')
    parser.add_argument(
        '--out-val-path',
        default="annotations/val.json",
        help='Path to save val json file')
    parser.add_argument(
        '--out-test-path',
        default="annotations/test.json",
        help='Path to save test json file')
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


def get_categories():
    classes_names = ["object"]
    return {name: i for i, name in enumerate(classes_names)}


def convert_n_save(df, json_file):
    json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    categories = get_categories()
    bnd_id = START_BOUNDING_BOX_ID
    for name, group in tqdm(df.groupby("filename")):
        filename = name
        image_id = get_filename_as_int(filename)
        width = int(group.w.iloc[0])
        height = int(group.h.iloc[0])
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        for row in group.itertuples():
            category = "object"
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            xmin = int(row.x1)
            ymin = int(row.y1)
            xmax = int(row.x2)
            ymax = int(row.y2)
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


def main():
    args = parse_args()

    train_df = pd.read_csv(args.in_train_path, header=None)
    val_df = pd.read_csv(args.in_val_path, header=None)
    test_df = pd.read_csv(args.in_test_path, header=None)

    train_df.columns = COLUMNS
    val_df.columns = COLUMNS
    test_df.columns = COLUMNS

    convert_n_save(train_df, args.out_train_path)
    convert_n_save(val_df, args.out_val_path)
    convert_n_save(test_df, args.out_test_path)


if __name__ == '__main__':
    main()
