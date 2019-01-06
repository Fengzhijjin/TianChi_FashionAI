from PIL import Image
import csv
import numpy as np
import random

from util.tfrecord_tools import TFRecord

ATTRKEYS = {"coat_length_labels": 8, "collar_design_labels": 5,
            "lapel_design_labels": 5, "neck_design_labels": 5,
            "neckline_design_labels": 10, "pant_length_labels": 6,
            "skirt_length_labels": 6, "sleeve_length_labels": 9}

SEGS_LABEL = ["lapel_design_labels", "neck_design_labels",
              "neckline_design_labels", "collar_design_labels"]

TRAIN_DATA_FILE = "./data/train/"
TRAIN_SAVE_PATH = "./data/tfrecords/train/"
# TRAIN_SAVE_PATH = "/media/fengzhijin/Linux附加卷/Migration_Learning/迁移学习/data/tfrecords/train/"
TRAIN_LABEL_FILE = "./data/train/Annotations/label.csv"

VALI_SAVE_PATH = "./data/tfrecords/validation/"
# VALI_SAVE_PATH = "/media/fengzhijin/Linux附加卷/Migration_Learning/迁移学习/data/tfrecords/validation/"

ATTRVALUES = {'y': 1.0, 'm': 0.0, 'n': 0.0}


def read_data(file):
    with open(file, "r") as csvfile:
        read = csv.reader(csvfile)
        read = list(read)
        read = np.array(read)
    return read


def convert_to_train(name, data_file, label_file, save_file, val_file=VALI_SAVE_PATH):
    data = read_data(label_file)

    image_file = list(data[:, 0])
    attrkey = list(data[:, 1])
    label = list(data[:, 2])

    for i in range(len(label)):
        lab = list()
        for j in label[i]:
            lab.append(ATTRVALUES[j])
        label[i] = lab

    train_coat_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 8})
    # vali_coat_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 8})
    writer_coat_train = train_coat_tfrecords.writer(
        save_dir=save_file + "coat_length_labels/", prefix=name)
    # writer_coat_vali = vali_coat_tfrecords.writer(
    #     save_dir=val_file + "coat_length_labels/", prefix="validation")

    train_collar_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 5})
    # vali_collar_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 5})
    writer_collar_train = train_collar_tfrecords.writer(
        save_dir=save_file + "collar_design_labels/", prefix=name)
    # writer_collar_vali = vali_collar_tfrecords.writer(
    #     save_dir=val_file + "collar_design_labels/", prefix="validation")

    train_lapel_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 5})
    # vali_lapel_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 5})
    writer_lapel_train = train_lapel_tfrecords.writer(
        save_dir=save_file + "lapel_design_labels/", prefix=name)
    # writer_lapel_vali = vali_lapel_tfrecords.writer(
    #     save_dir=val_file + "lapel_design_labels/", prefix="validation")

    train_neck_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 5})
    # vali_neck_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 5})
    writer_neck_train = train_neck_tfrecords.writer(
        save_dir=save_file + "neck_design_labels/", prefix=name)
    # writer_neck_vali = vali_neck_tfrecords.writer(
    #     save_dir=val_file + "neck_design_labels/", prefix="validation")

    train_neckline_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 10})
    # vali_neckline_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 10})
    writer_neckline_train = train_neckline_tfrecords.writer(
        save_dir=save_file + "neckline_design_labels/", prefix=name)
    # writer_neckline_vali = vali_neckline_tfrecords.writer(
    #     save_dir=val_file + "neckline_design_labels/", prefix="validation")

    train_pant_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 6})
    # vali_pant_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 6})
    writer_pant_train = train_pant_tfrecords.writer(
        save_dir=save_file + "pant_length_labels/", prefix=name)
    # writer_pant_vali = vali_pant_tfrecords.writer(
    #     save_dir=val_file + "pant_length_labels/", prefix="validation")

    train_skirt_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 6})
    # vali_skirt_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 6})
    writer_skirt_train = train_skirt_tfrecords.writer(
        save_dir=save_file + "skirt_length_labels/", prefix=name)
    # writer_skirt_vali = vali_skirt_tfrecords.writer(
    #     save_dir=val_file + "skirt_length_labels/", prefix="validation")

    train_sleeve_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 9})
    # vali_sleeve_tfrecords = TFRecord({'img': [bytes], 'label': [float] * 9})
    writer_sleeve_train = train_sleeve_tfrecords.writer(
        save_dir=save_file + "sleeve_length_labels/", prefix=name)
    # writer_sleeve_vali = vali_sleeve_tfrecords.writer(
    #     save_dir=val_file + "sleeve_length_labels/", prefix="validation")

    coat_num, collar_num, lapel_num, neck_num, neckline_num, pant_num, skirt_num, sleeve_num = 0, 0, 0, 0, 0, 0, 0, 0

    for i in range(len(image_file)):
        print(image_file[i], label[i])
        img = Image.open(data_file + image_file[i])
        if attrkey[i] in SEGS_LABEL:
            box = (img.size[0] / 6, 0, img.size[0] * 5 / 6, img.size[1] * 2 / 3)
            img = img.crop(box)
        img = img.resize((224, 224))
        img_1 = img.transpose(Image.FLIP_LEFT_RIGHT)
        rotate = random.randint(-15, 15)
        img_2 = img.rotate(rotate)
        img = img.tobytes()
        img_1 = img_1.tobytes()
        img_2 = img_2.tobytes()
        if attrkey[i] == 'coat_length_labels':
            if coat_num % 10 == 0:
                # writer_coat_vali.add_example({'img': [img], 'label': label[i]})
                writer_coat_train.add_example({'img': [img], 'label': label[i]})
                writer_coat_train.add_example({'img': [img_1], 'label': label[i]})
                writer_coat_train.add_example({'img': [img_2], 'label': label[i]})
            else:
                writer_coat_train.add_example({'img': [img], 'label': label[i]})
                writer_coat_train.add_example({'img': [img_1], 'label': label[i]})
                writer_coat_train.add_example({'img': [img_2], 'label': label[i]})
            coat_num += 1
        elif attrkey[i] == 'collar_design_labels':
            if collar_num % 10 == 0:
                # writer_collar_vali.add_example({'img': [img], 'label': label[i]})
                writer_collar_train.add_example({'img': [img], 'label': label[i]})
                writer_collar_train.add_example({'img': [img_1], 'label': label[i]})
                writer_collar_train.add_example({'img': [img_2], 'label': label[i]})
            else:
                writer_collar_train.add_example({'img': [img], 'label': label[i]})
                writer_collar_train.add_example({'img': [img_1], 'label': label[i]})
                writer_collar_train.add_example({'img': [img_2], 'label': label[i]})
            collar_num += 1
        elif attrkey[i] == 'lapel_design_labels':
            if lapel_num % 10 == 0:
                # writer_lapel_vali.add_example({'img': [img], 'label': label[i]})
                writer_lapel_train.add_example({'img': [img], 'label': label[i]})
                writer_lapel_train.add_example({'img': [img_1], 'label': label[i]})
                writer_lapel_train.add_example({'img': [img_2], 'label': label[i]})
            else:
                writer_lapel_train.add_example({'img': [img], 'label': label[i]})
                writer_lapel_train.add_example({'img': [img_1], 'label': label[i]})
                writer_lapel_train.add_example({'img': [img_2], 'label': label[i]})
            lapel_num += 1
        elif attrkey[i] == 'neck_design_labels':
            if neck_num % 10 == 0:
                # writer_neck_vali.add_example({'img': [img], 'label': label[i]})
                writer_neck_train.add_example({'img': [img], 'label': label[i]})
                writer_neck_train.add_example({'img': [img_1], 'label': label[i]})
                writer_neck_train.add_example({'img': [img_2], 'label': label[i]})
            else:
                writer_neck_train.add_example({'img': [img], 'label': label[i]})
                writer_neck_train.add_example({'img': [img_1], 'label': label[i]})
                writer_neck_train.add_example({'img': [img_2], 'label': label[i]})
            neck_num += 1
        elif attrkey[i] == 'neckline_design_labels':
            if neckline_num % 10 == 0:
                # writer_neckline_vali.add_example({'img': [img], 'label': label[i]})
                writer_neckline_train.add_example({'img': [img], 'label': label[i]})
                writer_neckline_train.add_example({'img': [img_1], 'label': label[i]})
                writer_neckline_train.add_example({'img': [img_2], 'label': label[i]})
            else:
                writer_neckline_train.add_example({'img': [img], 'label': label[i]})
                writer_neckline_train.add_example({'img': [img_1], 'label': label[i]})
                writer_neckline_train.add_example({'img': [img_2], 'label': label[i]})
            neckline_num += 1
        elif attrkey[i] == 'pant_length_labels':
            if pant_num % 10 == 0:
                # writer_pant_vali.add_example({'img': [img], 'label': label[i]})
                writer_pant_train.add_example({'img': [img], 'label': label[i]})
                writer_pant_train.add_example({'img': [img_1], 'label': label[i]})
                writer_pant_train.add_example({'img': [img_2], 'label': label[i]})
            else:
                writer_pant_train.add_example({'img': [img], 'label': label[i]})
                writer_pant_train.add_example({'img': [img_1], 'label': label[i]})
                writer_pant_train.add_example({'img': [img_2], 'label': label[i]})
            pant_num += 1
        elif attrkey[i] == 'skirt_length_labels':
            if skirt_num % 10 == 0:
                # writer_skirt_vali.add_example({'img': [img], 'label': label[i]})
                writer_skirt_train.add_example({'img': [img], 'label': label[i]})
                writer_skirt_train.add_example({'img': [img_1], 'label': label[i]})
                writer_skirt_train.add_example({'img': [img_2], 'label': label[i]})
            else:
                writer_skirt_train.add_example({'img': [img], 'label': label[i]})
                writer_skirt_train.add_example({'img': [img_1], 'label': label[i]})
                writer_skirt_train.add_example({'img': [img_2], 'label': label[i]})
            skirt_num += 1
        if attrkey[i] == 'sleeve_length_labels':
            if sleeve_num % 10 == 0:
                # writer_sleeve_vali.add_example({'img': [img], 'label': label[i]})
                writer_sleeve_train.add_example({'img': [img], 'label': label[i]})
                writer_sleeve_train.add_example({'img': [img_1], 'label': label[i]})
                writer_sleeve_train.add_example({'img': [img_2], 'label': label[i]})
            else:
                writer_sleeve_train.add_example({'img': [img], 'label': label[i]})
                writer_sleeve_train.add_example({'img': [img_1], 'label': label[i]})
                writer_sleeve_train.add_example({'img': [img_2], 'label': label[i]})
            sleeve_num += 1
    writer_coat_train.close()
    writer_collar_train.close()
    writer_lapel_train.close()
    writer_neck_train.close()
    writer_neckline_train.close()
    writer_pant_train.close()
    writer_skirt_train.close()
    writer_sleeve_train.close()
    # writer_coat_vali.close()
    # writer_collar_vali.close()
    # writer_lapel_vali.close()
    # writer_neck_vali.close()
    # writer_neckline_vali.close()
    # writer_pant_vali.close()
    # writer_skirt_vali.close()
    # writer_sleeve_vali.close()
    return 0


if __name__ == "__main__":
    convert_to_train('train', TRAIN_DATA_FILE, TRAIN_LABEL_FILE, TRAIN_SAVE_PATH)
