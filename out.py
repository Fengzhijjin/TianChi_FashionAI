import tensorflow as tf
import csv
import numpy as np
from PIL import Image


TEST_DATA_FILE = "./data/test/"
TEST_LABEL_FILE = "./data/test/Tests/question.csv"

CATEGORY_KEYS = ["collar_design_labels", "neckline_design_labels",
                 "skirt_length_labels", "sleeve_length_labels",
                 "neck_design_labels", "coat_length_labels",
                 "lapel_design_labels", "pant_length_labels"]

CATEGORY_NUMBER = {"collar_design_labels": [0, 1591], "neckline_design_labels": [1591, 4738],
                   "skirt_length_labels": [4738, 6493], "sleeve_length_labels": [6493, 9027],
                   "neck_design_labels": [9027, 10118], "coat_length_labels": [10118, 12265],
                   "lapel_design_labels": [12265, 13608], "pant_length_labels": [13608, 15042]}

SEGS_LABEL = ["lapel_design_labels", "neck_design_labels",
              "neckline_design_labels", "collar_design_labels"]

TEST_SIZE = 15042


def read_data(file):
    with open(file, "r") as csvfile:
        read = csv.reader(csvfile)
        read = list(read)
        read = np.array(read)
    return read


def read_data_test(name, start, end):
    data = read_data(TEST_LABEL_FILE)

    image_file = list(data[:, 0])

    for i in range(start, end):
        print(image_file[i])
        img = Image.open(TEST_DATA_FILE + image_file[i])
        if name in SEGS_LABEL:
            box = (img.size[0] / 6, 0, img.size[0] * 5 / 6, img.size[1] * 2 / 3)
            img = img.crop(box)
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        img = img.reshape((-1, 224, 224, 3))
        if i == start:
            image = img
        else:
            image = np.concatenate([image, img])
    return image


def train_0(data, name):
    with tf.Graph().as_default() as new_graph:
        x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
        with tf.gfile.FastGFile('./model/' + name + '/graph.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_out = tf.import_graph_def(
                graph_def,
                input_map={'x-input:0': x},
                return_elements=['out5/Sigmoid:0'])

    with tf.Session(graph=new_graph) as sess:
        return sess.run(g_out, feed_dict={x: data})


def train_1(data, name):
    with tf.Graph().as_default() as new_graph:
        x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
        with tf.gfile.FastGFile('./model_bak1/' + name + '/graph.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_out = tf.import_graph_def(
                graph_def,
                input_map={'x-input:0': x},
                return_elements=['out5/Sigmoid:0'])

    with tf.Session(graph=new_graph) as sess:
        return sess.run(g_out, feed_dict={x: data})


def train_2(data, name):
    with tf.Graph().as_default() as new_graph:
        x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
        with tf.gfile.FastGFile('./model_bak2/' + name + '/graph.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_out = tf.import_graph_def(
                graph_def,
                input_map={'x-input:0': x},
                return_elements=['out5/Sigmoid:0'])

    with tf.Session(graph=new_graph) as sess:
        return sess.run(g_out, feed_dict={x: data})


if __name__ == "__main__":

    test = read_data(TEST_LABEL_FILE)
    image_file = list(test[:, 0])
    image_array = list(test[:, 1])
    image_labels = list()

    for i in CATEGORY_KEYS:
        teX = read_data_test(i, CATEGORY_NUMBER[i][0], CATEGORY_NUMBER[i][1])
        batch = 32
        batch_num = len(teX) // batch
        for j in range(batch_num + 1):
            if j == 0:
                labels_0 = train_0(teX[batch * j:batch * (j + 1)], i)
                labels_1 = train_1(teX[batch * j:batch * (j + 1)], i)
                labels_2 = train_2(teX[batch * j:batch * (j + 1)], i)
            elif j == batch_num:
                labels_0 = np.concatenate([labels_0, train_0(teX[batch * j:], i)],
                                          axis=1)
                labels_1 = np.concatenate([labels_1, train_1(teX[batch * j:], i)],
                                          axis=1)
                labels_2 = np.concatenate([labels_2, train_2(teX[batch * j:], i)],
                                          axis=1)
            else:
                labels_0 = np.concatenate([labels_0, train_0(teX[batch * j:batch * (j + 1)], i)],
                                          axis=1)
                labels_1 = np.concatenate([labels_1, train_0(teX[batch * j:batch * (j + 1)], i)],
                                          axis=1)
                labels_2 = np.concatenate([labels_2, train_0(teX[batch * j:batch * (j + 1)], i)],
                                          axis=1)
        print("-----------------------------------")
        print(np.shape(labels_0))
        print(np.shape(labels_1))
        print(np.shape(labels_2))
        for i in range(len(labels_0[0])):
            l = (labels_0[0, i] + labels_1[0, i] + labels_2[0, i]) / 3.0
            # l = labels_0[0, i]
            label = ""
            for j in l:
                label = label + str(j) + ';'
            image_labels.append(label[:-1])
    print(len(image_file))
    print(len(image_array))
    print(len(image_labels))
    with open("./data/test/Tests/test_01_ResNet50.csv", "w") as datacsv:
        csvwriter = csv.writer(datacsv, delattr)
        for i in range(1000):
            csvwriter.writerow([image_file[i], image_array[i], image_labels[i]])
