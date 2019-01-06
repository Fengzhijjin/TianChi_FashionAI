# coding:utf-8
import nets.resnet_v2 as resnet_v2
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
from tensorflow.python.framework import graph_util


NAN = ["collar_design_labels", "lapel_design_labels",
       "neck_design_labels", "neckline_design_labels"]

AI = {}


def loss_func(py_x, Y):
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    # loss = tf.reduce_mean(tf.square(py_x - Y), axis=0)
    # loss = tf.reduce_sum(loss)
    loss = tf.reduce_mean(tf.square(py_x - Y), axis=0)
    loss = tf.reduce_sum(loss)
    return loss


def conv2d_bn(input_data, kernel_num, filter_size=[2, 2], stride=1, padding='VALID',
              is_training=True, variables_collections=None, scope=None):
    res = slim.conv2d(
        input_data,
        kernel_num,
        filter_size,
        stride=stride,
        padding=padding,
        activation_fn=tf.nn.relu,
        variables_collections=variables_collections,
        scope=scope)
    res = slim.batch_norm(res, is_training=is_training, scope=scope)
    return res


def read_data(filename_queue, label_num):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([label_num], tf.float32)
        })
    image = tf.decode_raw(features['img'], tf.uint8)
    label = features['label']
    image = tf.reshape(image, [224, 224, 3])
    image = tf.cast(image, tf.float32) / tf.constant(255.)
    return image, label


def get_data_train(name, file, batch=256):
    filename_queue = tf.train.string_input_producer(file)
    if name == 'coat_length_labels':
        label_num = 8
    elif name == 'collar_design_labels':
        label_num = 5
    elif name == 'lapel_design_labels':
        label_num = 5
    elif name == 'neck_design_labels':
        label_num = 5
    elif name == 'neckline_design_labels':
        label_num = 10
    elif name == 'pant_length_labels':
        label_num = 6
    elif name == 'skirt_length_labels':
        label_num = 6
    elif name == 'sleeve_length_labels':
        label_num = 9
    image, label = read_data(filename_queue, label_num)
    out_image, out_label = tf.train.shuffle_batch([image, label],
                                                  batch_size=batch,
                                                  capacity=batch * 3,
                                                  min_after_dequeue=batch * 2)
    return out_image, out_label


def get_data_validation(name, file, batch=256):
    filename_queue = tf.train.string_input_producer(file)
    if name == 'coat_length_labels':
        label_num = 8
    elif name == 'collar_design_labels':
        label_num = 5
    elif name == 'lapel_design_labels':
        label_num = 5
    elif name == 'neck_design_labels':
        label_num = 5
    elif name == 'neckline_design_labels':
        label_num = 10
    elif name == 'pant_length_labels':
        label_num = 6
    elif name == 'skirt_length_labels':
        label_num = 6
    elif name == 'sleeve_length_labels':
        label_num = 9
    image, label = read_data(filename_queue, label_num)
    out_image, out_label = tf.train.batch([image, label],
                                          batch_size=batch,
                                          capacity=batch * 3)
    return out_image, out_label


def train(name):
    with tf.Graph().as_default() as graph:
        if name == 'coat_length_labels':
            train_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/train/coat_length_labels/", '*.tfrecord'))
            test_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/validation/coat_length_labels/", '*.tfrecord'))
            save_file_path = "./model/coat_length_labels/"
            X = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x-input')
            Y = tf.placeholder(tf.float32, [None, 8], name='y-input')
            num = 8
        elif name == 'collar_design_labels':
            train_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/train/collar_design_labels/", '*.tfrecord'))
            test_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/validation/collar_design_labels/", '*.tfrecord'))
            save_file_path = "./model/collar_design_labels/"
            X = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x-input')
            Y = tf.placeholder(tf.float32, [None, 5], name='y-input')
            num = 5
        elif name == 'lapel_design_labels':
            train_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/train/lapel_design_labels/", '*.tfrecord'))
            test_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/validation/lapel_design_labels/", '*.tfrecord'))
            save_file_path = "./model/lapel_design_labels/"
            X = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x-input')
            Y = tf.placeholder(tf.float32, [None, 5], name='y-input')
            num = 5
        elif name == 'neck_design_labels':
            train_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/train/neck_design_labels/", '*.tfrecord'))
            test_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/validation/neck_design_labels/", '*.tfrecord'))
            save_file_path = "./model/neck_design_labels/"
            X = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x-input')
            Y = tf.placeholder(tf.float32, [None, 5], name='y-input')
            num = 5
        elif name == 'neckline_design_labels':
            train_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/train/neckline_design_labels/", '*.tfrecord'))
            test_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/validation/neckline_design_labels", '*.tfrecord'))
            save_file_path = "./model/neckline_design_labels/"
            X = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x-input')
            Y = tf.placeholder(tf.float32, [None, 10], name='y-input')
            num = 10
        elif name == 'pant_length_labels':
            train_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/train/pant_length_labels/", '*.tfrecord'))
            test_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/validation/pant_length_labels/", '*.tfrecord'))
            save_file_path = "./model/pant_length_labels/"
            X = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x-input')
            Y = tf.placeholder(tf.float32, [None, 6], name='y-input')
            num = 6
        elif name == 'skirt_length_labels':
            train_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/train/skirt_length_labels/", '*.tfrecord'))
            test_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/validation/skirt_length_labels/", '*.tfrecord'))
            save_file_path = "./model/skirt_length_labels/"
            X = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x-input')
            Y = tf.placeholder(tf.float32, [None, 6], name='y-input')
            num = 6
        elif name == 'sleeve_length_labels':
            train_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/train/sleeve_length_labels/", '*.tfrecord'))
            test_file_path = tf.gfile.Glob(os.path.join("./data/tfrecords/validation/sleeve_length_labels/", '*.tfrecord'))
            save_file_path = "./model/sleeve_length_labels/"
            X = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x-input')
            Y = tf.placeholder(tf.float32, [None, 9], name='y-input')
            num = 9

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            trX_, trY_ = get_data_train(name, train_file_path, 256)
            tvX_, tvY_ = get_data_validation(name, test_file_path, 1024)
            net, end_points = resnet_v2.resnet_v2_50(X, is_training=False, global_pool=False)
            net = end_points['resnet_v2_50/block4/unit_1/bottleneck_v2/conv1']
            net = conv2d_bn(net, 512, [3, 3], variables_collections=['CusW'], scope='out1')
            net = conv2d_bn(net, 512, [3, 3], variables_collections=['CusW'], scope='out2')
            net = conv2d_bn(net, 512, [3, 3], variables_collections=['CusW'], scope='out3')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, variables_collections=['CusW'], scope='out4')
            net = slim.fully_connected(net, num, activation_fn=tf.nn.sigmoid, variables_collections=['CusW'], scope='out5')
            cost = loss_func(net, Y)
            var_list = [*tf.get_collection('CusW')]
            print(net)
            optimizer = tf.train.AdamOptimizer()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([tf.group(*update_ops)]):
                train_op = optimizer.minimize(cost, var_list=var_list)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        print("恢复模型:")
        model_path = './resnet_v2_50_2017_04_14/resnet_v2_50.ckpt'
        variables_to_restore = slim.get_variables_to_restore(
            exclude=['out1', 'out2', 'out3', 'out4', 'out5']
        )
        init_fn = slim.assign_from_checkpoint_fn(
            model_path, variables_to_restore, ignore_missing_vars=True)
        init_fn(sess)
        print("模型恢复成功")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        tvX, tvY = sess.run([tvX_, tvY_])
        cost_min = 100.0
        cost_num = 0
        cost_i = 0
        begin = datetime.datetime.now()
        for i in range(1, 1001):
            count_s = 0.0
            cost_ = 0.0
            trX, trY = sess.run([trX_, trY_])
            training_batch = zip(range(0, len(trX), 64),
                                 range(64, len(trX)+1, 64))
            for start, end in training_batch:
                cost_train, train_op_ = sess.run(
                    [cost, train_op],
                    feed_dict={X: trX[start:end], Y: trY[start:end]})

            if i % 10 == 0:
                valiing_batch = zip(range(0, len(tvX), 32),
                                    range(32, len(tvX)+1, 32))
                for start, end in valiing_batch:
                    vali_op_, cost_vali = sess.run(
                        [net, cost],
                        feed_dict={X: tvX[start:end], Y: tvY[start:end]})

                    if start == 0:
                        cost_ = cost_vali
                    else:
                        cost_ = (cost_vali + cost_) / 2

                    vali_op = np.argmax(vali_op_, 1)
                    vali_y = np.argmax(tvY[start:end], 1)
                    if start == 0:
                        count_s = np.mean(vali_op == vali_y)
                    else:
                        count_s = (count_s + np.mean(vali_op == vali_y)) / 2
                end = datetime.datetime.now()
                print(i, count_s, cost_, end-begin)
                begin = datetime.datetime.now()

                if cost_ < cost_min:
                    cost_num = 0
                    cost_min = cost_
                    cost_i = i
                    new_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['out5/Sigmoid'])
                    tf.train.write_graph(new_graph, save_file_path, 'graph.pb', as_text=False)
                    AI[name] = [cost_i, count_s]
                else:
                    cost_num += 1
                if cost_num == 10:
                    break
        coord.request_stop()
        coord.join(threads)
        print("迭代结束")
        print(cost_i, cost_min)


if __name__ == "__main__":
    lei = ["coat_length_labels", "collar_design_labels", "lapel_design_labels",
           "neck_design_labels", "neckline_design_labels", "pant_length_labels",
           "skirt_length_labels", "sleeve_length_labels"]

    for i in lei[0:]:
        print(i)
        train(i)
    print(AI)
