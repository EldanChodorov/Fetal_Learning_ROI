#!/usr/bin/python

'''
Run over all files which are .npz,
and if they have 3 arrays then open and save as RGB in an npz.
'''


import os
import imageio
import numpy as np

DATA = "data"
LABELS = "labels"

def test_img_shape(path):
    img = imageio.imread(path)
    print("==== > test path {}, shape: {}".format(path, img.shape))


def handle_file(path, full_path, new_path):
    # print("==> Handle file: {}".format(path))
    obj = np.load(full_path)
    if len(obj.keys()) == 3:
        arr_0, arr_1, arr_2 = obj['arr_0'], obj['arr_1'], obj['arr_2']
        if "gt" in full_path:
            new_arr = arr_1
        else:
            new_arr = np.zeros((arr_0.shape[0], arr_0.shape[1], 3))
            new_arr[:, :, 0] = arr_0
            new_arr[:, :, 1] = arr_1
            new_arr[:, :, 2] = arr_2
        imageio.imsave(new_path, new_arr)
        # print("     Saved to {}".format(full_path))


def preprocess_files(path, new_path):
    for f in os.listdir(data_path):
        # assume all files are .npz
        full_path = os.path.join(path, f)
        new_f = os.path.join(new_path, f)
        if not os.path.exists(new_f):
            handle_file(f, full_path, new_f)


def validate_test_gt():
    test_img_path = os.getcwd() + "\\test\\img\\0"
    test_gt_path = os.getcwd() + "\\test\\gt\\0"
    train_gt_path = os.getcwd() + "\\train\\gt\\0"
    for f in os.listdir(test_img_path):
        gt_new_path = os.path.join(test_gt_path, f)
        if not os.path.exists(gt_new_path):
            old_path = os.path.join(train_gt_path, f)
            os.rename(old_path, gt_new_path)


def main_preprocess():
    root_path = os.getcwd()
    data_root_path = "C:\\Users\\Eldan\\Documents\\Final Project\\AutomatedFetal_CV_project\\roi_ground_truth"

    data_path = os.path.join(data_root_path, DATA)
    data_new_path = os.path.join(root_path, "train", "img", "0")
    preprocess_files(data_path, data_new_path)

    labels_path = os.path.join(data_root_path, LABELS)
    labels_new_path = os.path.join(root_path, "train", "gt", "0")
    preprocess_files(labels_path, labels_new_path)


if __name__ == '__main__':
    # main_preprocess()
    # validate_test_gt()
    print('empty')
