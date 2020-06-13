import argparse
import json
from copy import deepcopy
from operator import itemgetter

import pandas as pd
from sklearn.model_selection import KFold


def parse_args():
    parser = argparse.ArgumentParser(
        description='Combine train, val and test parts of the \
        COCO-like dataset and split them on 5 train-test folds')
    parser.add_argument(
        'ann_dir_path',
        default="data/annotations",
        help='Path to the directory where current annotations are stored')
    parser.add_argument(
        '--save-whole',
        action='store_true',
        help='Whether to save whole dataset')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(f"{args.ann_dir_path}/train.json") as f:
        train = json.load(f)
    with open(f"{args.ann_dir_path}/val.json") as f:
        val = json.load(f)
    with open(f"{args.ann_dir_path}/test.json") as f:
        test = json.load(f)

    print(
        f"Images in train: {len(train['images'])}, val: {len(val['images'])}, test: {len(test['images'])}"
    )

    # shift ids in val dataset
    max_train_img_id = max([x['id'] for x in train["images"]])
    max_train_ann_id = max([x['id'] for x in train["annotations"]])
    for x in val['images']:
        x['id'] += max_train_img_id + 1

    for x in val['annotations']:
        x['image_id'] += max_train_img_id + 1
        x['id'] += max_train_ann_id + 1

    # shift ids in test dataset
    max_val_img_id = max([x['id'] for x in val["images"]])
    max_val_ann_id = max([x['id'] for x in val["annotations"]])
    for x in test['images']:
        x['id'] += max_val_img_id + 1

    for x in test['annotations']:
        x['image_id'] += max_val_img_id + 1
        x['id'] += max_val_ann_id + 1

    # combine them all
    train["images"].extend(val["images"])
    train["images"].extend(test["images"])
    train["annotations"].extend(val["annotations"])
    train["annotations"].extend(test["annotations"])

    # split
    folds_train = []
    folds_test = []
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in k_fold.split(train["images"]):
        train_images = itemgetter(*list(train_idx))(train["images"])
        test_images = itemgetter(*list(test_idx))(train["images"])
        train_images_ids = set([x["id"] for x in train_images])
        test_images_ids = set([x["id"] for x in test_images])
        train_anns = [
            x for x in train["annotations"]
            if x["image_id"] in train_images_ids
        ]
        test_anns = [
            x for x in train["annotations"] if x["image_id"] in test_images_ids
        ]

        fold_train = deepcopy(train)
        fold_test = deepcopy(train)

        fold_train["images"] = train_images
        fold_test["images"] = test_images
        fold_train["annotations"] = train_anns
        fold_test["annotations"] = test_anns

        folds_train.append(fold_train)
        folds_test.append(fold_test)

    for fold_n, (train_, test_) in enumerate(zip(folds_train, folds_test)):

        print(
            f"Fold: {fold_n}, train images: {len(train_['images'])}, \
              test images: {len(test_['images'])}"
        )
        print(
            f"train annotations: {len(train_['annotations'])}, \
              test annotations {len(test_['annotations'])}"
        )

        with open(f"{args.ann_dir_path}/train_{fold_n}.json", "w") as f:
            json.dump(train_, f)
        with open(f"{args.ann_dir_path}/val_{fold_n}.json", "w") as f:
            json.dump(test_, f)

    if args.save_whole:
        with open(f"{args.ann_dir_path}/train_whole.json", "w") as f:
            json.dump(train, f)


if __name__ == '__main__':
    main()
