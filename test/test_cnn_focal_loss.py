from model import *
import configparser as cp
import sys
import time
import os
from LoadData import LoadData
import netModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from progress.bar import Bar
import logging

name = "iccad1_config"

file_handler = logging.FileHandler("../log/" + name + "_test.log", mode="a", encoding="utf-8")
logger = logging.getLogger(name + " test with focal_loss ")
logging.basicConfig(level="INFO")
file_fmt = "%(name)s--->%(levelname)s--->%(asctime)s--->%(message)s"

fmt2 = logging.Formatter(fmt=file_fmt)

file_handler.setFormatter(fmt=fmt2)
logger.addHandler(file_handler)

'''
Initialize Path and Global Params
'''
infile = cp.SafeConfigParser()
# infile.read(sys.argv[1])
infile.read("../ini/"+name + ".ini")
test_path = infile.get('dir', 'test_path')
logger.info(test_path)
model_path = infile.get('dir', 'model_path')
logger.info(model_path)
fealen = int(infile.get('feature', 'ft_length'))
blockdim = int(infile.get('feature', 'block_dim'))
aug = int(infile.get('feature', 'AUG'))

'''
Prepare the Input
'''
test_data = LoadData(test_path, test_path + '/label.csv')
x_data = tf.placeholder(tf.float32, shape=[None, blockdim * blockdim, fealen])  # input FT
y_gt = tf.placeholder(tf.float32, shape=[None, 2])  # ground truth label
x = tf.reshape(x_data, [-1, blockdim, blockdim, fealen])  # reshap to NHWC
x_ud = tf.map_fn(lambda img: tf.image.flip_up_down(img), x)  # up down flipped
x_lr = tf.map_fn(lambda img: tf.image.flip_left_right(img), x)  # left right flipped
x_lu = tf.map_fn(lambda img: tf.image.flip_up_down(img), x_lr)  # both flipped
predict_or = netModel.forward_V1_SE(x, is_training=False)  # do forward
predict_ud = netModel.forward_V1_SE(x_ud, is_training=False, reuse=True)
predict_lr = netModel.forward_V1_SE(x_lr, is_training=False, reuse=True)
predict_lu = netModel.forward_V1_SE(x_lu, is_training=False, reuse=True)
if aug == 1:
    predict = (predict_or + predict_lr + predict_lu + predict_ud) / 4.0
else:
    predict = predict_or
# predict = predict_or
y = tf.cast(tf.argmax(predict, 1), tf.int32)
accu = tf.equal(y, tf.cast(tf.argmax(y_gt, 1), tf.int32))  # calc batch accu
accu = tf.reduce_mean(tf.cast(accu, tf.float32))
'''
Start testing
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    chs = 0  # correctly predicted hs
    cnhs = 0  # correctly predicted nhs
    ahs = 0  # actual hs
    anhs = 0  # actual hs
    start = time.time()
    bar = Bar('Detecting', max=test_data.maxlen // 1000 + 1)
    for titr in range(0, test_data.maxlen // 1000 + 1):
        if not titr == test_data.maxlen // 1000:
            tbatch = test_data.nextbatch(1000, fealen)
        else:
            tbatch = test_data.nextbatch(test_data.maxlen - titr * 1000, fealen)
        tdata = tbatch[0]
        tlabel = tbatch[1]
        tmp_y = y.eval(feed_dict={x_data: tdata, y_gt: tlabel})
        tmp_label = np.argmax(tlabel, axis=1)
        tmp = tmp_label + tmp_y
        chs += sum(tmp == 2)
        cnhs += sum(tmp == 0)
        ahs += sum(tmp_label)
        anhs += sum(tmp_label == 0)
        bar.next()
    bar.finish()
    # print (chs, ahs, cnhs, anhs)
    format_str = ('正确预测热点数量：%d, 实际热点数量：%d，正确预测非热点数量：%d， 实际非热点数量：%d')
    # logger.info("正确预测热点数量：", chs , " 实际热点数量：", ahs, " 正确预测非热点数量：", cnhs, " 实际非热点数量：", anhs)
    logger.info(format_str % (chs, ahs, cnhs, anhs))
    if not ahs == 0:
        hs_accu = 1.0 * chs / ahs
    else:
        hs_accu = 0
    fs = anhs - cnhs
    end = time.time()
# print (ahs, anhs)
logger.info('Hotspot Detection Accuracy is %f' % hs_accu)
logger.info('False Alarm is %f' % fs)
logger.info('False Positive Rate is %.2f%%' % (float(fs) / anhs * 100))
logger.info('Test Runtime is %f seconds' % (end - start))




