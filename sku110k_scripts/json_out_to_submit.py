import json
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Conver formatted json output to submission csv')
    parser.add_argument(
        '--in-json', default="submit.bbox.json", help='Path to the json file')
    parser.add_argument(
        '--out-file', default="submit.csv", help='Output csv file')
    args = parser.parse_args()
    return args


def conv4subm(input_path, output_path, file_prefix):
    with open(input_path, "r") as f:
        j = json.load(f)
    test_df = pd.DataFrame(j)
    test_df["x1"] = test_df.bbox.apply(lambda x: x[0]).astype(int)
    test_df["y1"] = test_df.bbox.apply(lambda x: x[1]).astype(int)
    test_df["x2"] = test_df.bbox.apply(lambda x: x[0] + x[2]).astype(int)
    test_df["y2"] = test_df.bbox.apply(lambda x: x[1] + x[3]).astype(int)
    test_df["image_name"] = test_df.image_id.apply(
        lambda x: file_prefix + str(x) + ".jpg")
    test_df = test_df[["image_name", "x1", "y1", "x2", "y2", "score"]]
    test_df.to_csv(output_path, header=None, index=False)


def main():
    args = parse_args()
    conv4subm(args.in_json, args.out_file, file_prefix="test2020_")


if __name__ == '__main__':
    main()