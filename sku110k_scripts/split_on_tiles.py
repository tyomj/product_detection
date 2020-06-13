import argparse
import json
import os

import cv2
import numpy as np
from tiling import Box, Tile
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Based on coco-style json ann file \
            split the images and thier bbox annotations on tiles')
    parser.add_argument(
        '--h-tiles', type=int, default=2, help='Number of horizontal tiles')
    parser.add_argument(
        '--w-tiles', type=int, default=2, help='Number of vertical tiles')
    parser.add_argument(
        '--box-crop-threshold',
        type=float,
        default=0.2,
        help='Include annotation only for boxes \
            which preserve more then this value on tile image')
    parser.add_argument(
        '--tiles-overlap-ratio',
        type=float,
        default=0.2,
        help='Make tiles overlap by a corresponding factor from original dim')
    parser.add_argument(
        '--in-file', default="annotations/val.json", help='Output json file')
    parser.add_argument(
        '--out-file',
        default="split_annotations/val.json",
        help='Output json file')
    parser.add_argument(
        '--ori-imdir', default="images", help='Folder with original images')
    parser.add_argument(
        '--out-imdir', default="split_images", help='Folder for tiles')
    args = parser.parse_args()
    return args


def add_tile_num(path, suffix):
    root, ext = os.path.splitext(path)
    return root + suffix + ext


def xywh2xyxy(xywhs):
    xyxys = []
    for bbox in xywhs:
        x1, y1, w, h = bbox
        xyxy = [x1, y1, x1 + w, y1 + h]
        xyxys.append(xyxy)
    return xyxys


def imsave(filepath, img):
    cv2.imwrite(filepath, img)


def get_crop(image, box):
    return image[box.y1:box.y2, box.x1:box.x2, :].copy()


def split_image(img, bboxes, **kwargs):
    tiles = Tile(Box(0, 0, img.shape[1], img.shape[0]), bboxes).split(**kwargs)

    img_tiles = []
    for tile in tiles:
        tile.to_ints().collapse(dc_box=0, dc_nested=1)
        img_tile = get_crop(img, tile.box)
        tile.move_to_origin()
        img_tiles.append(img_tile)

    return zip(img_tiles, tiles)


def split_on_tiles(img, bboxes, **kwargs):

    h_img, w_img = img.shape[:2]
    h_step = int(h_img / kwargs["h_tiles"])
    w_step = int(w_img / kwargs["w_tiles"])
    h_overlap = int(h_step * kwargs.get("tiles_overlap_ratio", 0))
    w_overlap = int(w_step * kwargs.get("tiles_overlap_ratio", 0))
    kwargs["overlap"] = (w_overlap, h_overlap)
    kwargs["size"] = (w_step + w_overlap, h_step + h_overlap)

    return split_image(img, bboxes, **kwargs)


def get_annotation(image_id, annotation_id, coordinates):
    # Count area
    x1, y1, x2, y2 = coordinates
    w = x2 - x1
    h = y2 - y1
    area = w * h
    # bbox to x,y,w,h coco format
    bbox = [x1, y1, w, h]
    # check boxes
    if (min(bbox) < 0) or (area <= 0):
        return
    return dict(
        area=area,
        bbox=bbox,
        category_id=0,
        id=annotation_id,
        image_id=image_id,
        iscrowd=0,
        segmentation=[],
    )


def main():
    args = parse_args()
    last_img_idx = 0
    last_ann_idx = 0

    # load original ann file
    with open(args.in_file, 'r') as f:
        ann_json = json.load(f)

    # create empty coco-like dataset
    coco_dataset_dict = {
        'type': 'type',
        'categories': [{
            'supercategory': 'none',
            'id': 0,
            'name': 'object'
        }],
        'images': [],
        'annotations': []
    }

    # split original images and anns on tiles
    for image in tqdm(ann_json['images']):
        img = cv2.imread(os.path.join(args.ori_imdir, image['file_name']))
        img_anns = [
            x for x in ann_json['annotations'] if x['image_id'] == image['id']
        ]
        bboxes_xyxy = xywh2xyxy([x['bbox'] for x in img_anns])
        bboxes = [
            Box(*coors, data=dict(category_id=0)) for coors in bboxes_xyxy
        ]

        # get tiles
        tiles = split_on_tiles(img, bboxes, **args)
        tiles_counter = 0
        for tile_img, tile_ann in tiles:
            dst_img_path = os.path.join(args.out_imdir, image['file_name'])
            dst_img_full_path = add_tile_num(dst_img_path,
                                             "_p" + str(tiles_counter))
            file_name = add_tile_num(image['file_name'],
                                     "_p" + str(tiles_counter))

            if len(tile_ann.tile_boxes) == 0:
                print('skip')
                continue

            # append to images
            coco_dataset_dict["images"].append(
                dict(
                    license=1,
                    file_name=file_name,
                    height=tile_img.shape[0],
                    width=tile_img.shape[1],
                    id=last_img_idx,
                ))

            # append annotations
            for tile_box in tile_ann.tile_boxes:
                box_annotation = get_annotation(last_img_idx, last_ann_idx,
                                                tile_box.as_tuple())
                if not box_annotation:
                    print("No boxes")
                    continue
                coco_dataset_dict["annotations"].append(box_annotation)
                last_ann_idx += 1

            last_img_idx += 1
            tiles_counter += 1
            imsave(dst_img_full_path, tile_img)

    # save newly created ann file
    with open(args.out_file, 'w') as f:
        json.dump(coco_dataset_dict, f)


if __name__ == '__main__':
    main()
