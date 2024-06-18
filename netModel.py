import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
# from tensorflow.keras import layers, Model
# from tensorflow import keras
from tensorflow.keras.layers import Layer, Dense  # 导入需要使用的层
from tensorflow.keras import Model  # 导入 Model 类
import block
import logging

logger = logging.getLogger("train netModel: ")
logging.basicConfig(level="INFO")


def forward_V2(input, is_training=True, reuse=False, scope='model', flip=False):
    if flip == True:
        input = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input)
        input = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), input)

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, stride=1, padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0)):
            net = slim.conv2d(input, 16, [3, 3], scope='conv1_1')
            net = slim.conv2d(net, 16, [3, 3], scope='conv1_2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')

            # # 在第一个残差块之后添加残差块
            # net = residual_block(net, 16, 'res_block1')

            net = slim.conv2d(net, 32, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv2_2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')
            print("the net shape is: ", net.shape)
            net = slim.flatten(net)
            print("展平后： ", net.shape)
            w_init = tf.contrib.layers.xavier_initializer(uniform=False)
            net = slim.fully_connected(net, 250, activation_fn=tf.nn.relu, scope='fc1')
            net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
            predict = slim.fully_connected(net, 2, activation_fn=None, scope='fc2')
            print("预测输出形状：", predict.shape)
    return predict, tf.trainable_variables(scope)





def se_block(input, ratio=16):
    input_channels = int(input.get_shape()[-1])
    print("input_channels is: ", input_channels)
    squeeze = tf.reduce_mean(input, axis=[1, 2], keepdims=True)  # Global average pooling
    node = int(input_channels // ratio)
    excitation = slim.fully_connected(squeeze, node, activation_fn=tf.nn.relu)
    excitation = slim.fully_connected(excitation, input_channels, activation_fn=tf.nn.sigmoid)
    return input * excitation


def conv2d_with_bn(input, num_outputs, kernel_size, scope, stride=1):
    net = slim.conv2d(input, num_outputs, kernel_size, stride=stride, scope=scope)
    net = slim.batch_norm(net, scope=scope + '_bn')
    return net


def forward_V1_SE(input, is_training=True, reuse=False, scope='model', flip=False):
    if flip == True:
        input = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input)
        input = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), input)

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.leaky_relu, stride=1, padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0)):
            net = conv2d_with_bn(input, 16, [3, 3], scope='conv1_1')
            net = conv2d_with_bn(net, 16, [3, 3], scope='conv1_2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')

            # Add SE block after the first convolutional block
            net = se_block(net)

            net = conv2d_with_bn(net, 32, [3, 3], scope='conv2_1')
            net = conv2d_with_bn(net, 32, [3, 3], scope='conv2_2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')

            # Add SE block after the second convolutional block
            net = se_block(net)

            net = slim.flatten(net)
            net = slim.fully_connected(net, 250, activation_fn=tf.nn.relu, scope='fc1')
            net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
            predict = slim.fully_connected(net, 2, activation_fn=None, scope='fc2')
    return predict


def forward_ALexNet(input, is_training=True, reuse=False, scope='AlexNet'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.constant_initializer(0.1),
                            activation_fn=tf.nn.relu):
            net = slim.conv2d(input, 96, [11, 11], stride=4, padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
            net = slim.conv2d(net, 256, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool2')
            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool5')
            net = slim.flatten(net, scope='flatten5')
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.dropout(net, is_training=is_training, scope='dropout6')
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(net, is_training=is_training, scope='dropout7')
            predict = slim.fully_connected(net, 2, activation_fn=None, scope='fc8')
    return predict


# def conv2d_with_bn
def alexnet(input, is_training=True, reuse=False, scope='alexnet', flip=False):
    if flip == True:
        input = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input)
        input = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), input)

    with tf.variable_scope(scope, reuse=reuse):
        net = tf.contrib.layers.conv2d(input, num_outputs=96, kernel_size=11, stride=1, padding='VALID',
                                       activation_fn=tf.nn.relu, scope='conv1')
        net = tf.contrib.layers.max_pool2d(net, kernel_size=2, stride=2, padding='VALID', scope='pool1')
        net = tf.contrib.layers.conv2d(net, num_outputs=256, kernel_size=5, stride=1, padding='SAME',
                                       activation_fn=tf.nn.relu, scope='conv2')
        net = tf.contrib.layers.max_pool2d(net, kernel_size=1, stride=2, padding='VALID', scope='pool2')
        net = tf.contrib.layers.conv2d(net, num_outputs=384, kernel_size=3, stride=1, padding='SAME',
                                       activation_fn=tf.nn.relu, scope='conv3')
        net = tf.contrib.layers.conv2d(net, num_outputs=384, kernel_size=3, stride=1, padding='SAME',
                                       activation_fn=tf.nn.relu, scope='conv4')
        net = tf.contrib.layers.conv2d(net, num_outputs=256, kernel_size=3, stride=1, padding='SAME',
                                       activation_fn=tf.nn.relu, scope='conv5')
        net = tf.contrib.layers.max_pool2d(net, kernel_size=1, stride=2, padding='VALID', scope='pool5')
        net = tf.contrib.layers.flatten(net, scope='flatten')
        net = tf.contrib.layers.fully_connected(net, num_outputs=4096, activation_fn=tf.nn.relu, scope='fc6')
        net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout6')
        net = tf.contrib.layers.fully_connected(net, num_outputs=4096, activation_fn=tf.nn.relu, scope='fc7')
        net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout7')
        output = tf.contrib.layers.fully_connected(net, num_outputs=2, activation_fn=None, scope='fc8')

    return output


def se_block_V2(input, ratio=16):
    input_channels = int(input.get_shape()[-1])
    squeeze = tf.reduce_mean(input, axis=[1, 2], keepdims=True)  # Global average pooling
    node = int(input_channels // ratio)
    excitation = tf.layers.dense(squeeze, node, activation=tf.nn.relu)
    excitation = tf.layers.dense(excitation, input_channels, activation=tf.nn.sigmoid)
    return input * excitation


def alexnet_se(input, is_training=True, reuse=False, scope='alexnet', flip=False):
    logger.info(scope)
    if flip == True:
        input = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input)
        input = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), input)

    with tf.variable_scope(scope, reuse=reuse):
        net = tf.contrib.layers.conv2d(input, num_outputs=96, kernel_size=11, stride=1, padding='VALID',
                                       activation_fn=tf.nn.relu, scope='conv1')
        net = tf.contrib.layers.max_pool2d(net, kernel_size=2, stride=2, padding='VALID', scope='pool1')

        net = se_block(net)

        net = tf.contrib.layers.conv2d(net, num_outputs=256, kernel_size=5, stride=1, padding='SAME',
                                       activation_fn=tf.nn.relu, scope='conv2')
        net = tf.contrib.layers.max_pool2d(net, kernel_size=1, stride=2, padding='VALID', scope='pool2')

        net = se_block(net)

        net = tf.contrib.layers.conv2d(net, num_outputs=384, kernel_size=3, stride=1, padding='SAME',
                                       activation_fn=tf.nn.relu, scope='conv3')
        net = tf.contrib.layers.conv2d(net, num_outputs=384, kernel_size=3, stride=1, padding='SAME',
                                       activation_fn=tf.nn.relu, scope='conv4')
        net = tf.contrib.layers.conv2d(net, num_outputs=256, kernel_size=3, stride=1, padding='SAME',
                                       activation_fn=tf.nn.relu, scope='conv5')
        net = tf.contrib.layers.max_pool2d(net, kernel_size=1, stride=2, padding='VALID', scope='pool5')
        net = tf.contrib.layers.flatten(net, scope='flatten')
        net = tf.contrib.layers.fully_connected(net, num_outputs=4096, activation_fn=tf.nn.relu, scope='fc6')
        net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout6')
        net = tf.contrib.layers.fully_connected(net, num_outputs=4096, activation_fn=tf.nn.relu, scope='fc7')
        net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout7')
        output = tf.contrib.layers.fully_connected(net, num_outputs=2, activation_fn=None, scope='fc8')

    return output


def residual_block(inputs, num_filters, stride=1, scope='residual_block'):
    with tf.variable_scope(scope):
        residual = slim.conv2d(inputs, num_filters, [3, 3], stride=stride, activation_fn=None, scope='conv1')
        residual = tf.nn.relu(residual)
        residual = slim.conv2d(residual, num_filters, [3, 3], activation_fn=None, scope='conv2')
        if stride != 1 or inputs.get_shape()[3] != num_filters:
            shortcut = slim.conv2d(inputs, num_filters, [1, 1], stride=stride, activation_fn=None, scope='shortcut')
        else:
            shortcut = inputs
        output = tf.nn.relu(residual + shortcut)
    return output


def resnet(inputs, is_training=True, reuse=False, scope='alexnet', flip=False):
    if flip == True:
        inputs = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), inputs)
        inputs = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), inputs)
    with tf.variable_scope(scope, reuse=reuse):
        net = slim.conv2d(inputs, 64, [7, 7], stride=2, padding='SAME', activation_fn=None, scope='conv1')
        net = slim.batch_norm(net, is_training=is_training, scope='batch_norm1')
        net = tf.nn.relu(net)
        net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='max_pool')

        net = residual_block(net, 64, scope='residual_block1')
        net = residual_block(net, 64, scope='residual_block2')
        net = residual_block(net, 128, stride=2, scope='residual_block3')
        net = residual_block(net, 128, scope='residual_block4')
        net = residual_block(net, 256, stride=2, scope='residual_block5')
        net = residual_block(net, 256, scope='residual_block6')
        net = residual_block(net, 512, stride=2, scope='residual_block7')
        net = residual_block(net, 512, scope='residual_block8')

        net = slim.avg_pool2d(net, [1, 1], stride=1, padding='VALID', scope='avg_pool')
        net = slim.flatten(net, scope='flatten')
        net = slim.fully_connected(net, 2, activation_fn=None, scope='logits')

    return net


def GoogleNet(inputs, is_training=True, reuse=False, scope='InceptionV1', flip=False):
    if flip == True:
        inputs = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), inputs)
        inputs = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), inputs)
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='SAME'):
            net = slim.conv2d(inputs, 64, [7, 7], stride=2, padding='SAME', scope='conv1_7x7_s2')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='max_pool1_3x3_s2')

            net = slim.conv2d(net, 64, [1, 1], scope='conv2_3x3_reduce')
            net = slim.conv2d(net, 192, [3, 3], scope='conv2_3x3')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='max_pool2_3x3_s2')

        with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('inception_3a'):
                with tf.variable_scope('branch1x1'):
                    branch1x1 = slim.conv2d(net, 64, [1, 1], scope='conv1x1')
                with tf.variable_scope('branch5x5'):
                    branch5x5 = slim.conv2d(net, 48, [1, 1], scope='conv1x1')
                    branch5x5 = slim.conv2d(branch5x5, 64, [5, 5], scope='conv5x5')
                with tf.variable_scope('branch3x3dbl'):
                    branch3x3dbl = slim.conv2d(net, 64, [1, 1], scope='conv1x1')
                    branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3], scope='conv3x3')
                    branch3x3dbl = slim.conv2d(branch3x3dbl, 64, [3, 3], scope='conv3x3_2')
                with tf.variable_scope('branch_pool'):
                    branch_pool = slim.conv2d(net, 64, [1, 1], scope='conv1x1')
                net = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 3)

            with tf.variable_scope('inception_3b'):
                with tf.variable_scope('branch1x1'):
                    branch1x1 = slim.conv2d(net, 64, [1, 1], scope='conv1x1')
                with tf.variable_scope('branch5x5'):
                    branch5x5 = slim.conv2d(net, 48, [1, 1], scope='conv1x1')
                    branch5x5 = slim.conv2d(branch5x5, 64, [3, 3], scope='conv3x3')
                with tf.variable_scope('branch3x3dbl'):
                    branch3x3dbl = slim.conv2d(net, 64, [1, 1], scope='conv1x1')
                    branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3], scope='conv3x3')
                    branch3x3dbl = slim.conv2d(branch3x3dbl, 64, [3, 3], scope='conv3x3_2')
                with tf.variable_scope('branch_pool'):
                    branch_pool = slim.conv2d(net, 64, [1, 1], scope='conv1x1')
                net = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 3)

            # 最后添加一个全局平均池化层
            net = slim.avg_pool2d(net, [5, 5], padding='SAME', scope='avg_pool')

            # 将特征图扁平化
            net = slim.flatten(net)

            # 最后一个全连接层，输出大小为2
            net = slim.fully_connected(net, 2, activation_fn=None, scope='logits')

    return net


def channel_attention(input_feature, ratio=8):
    channel = input_feature.get_shape()[-1]
    shared_layer_one = tf.layers.dense(inputs=input_feature, units=channel // ratio, activation=tf.nn.relu)
    shared_layer_two = tf.layers.dense(inputs=shared_layer_one, units=channel, activation=tf.nn.relu)
    attention = tf.reduce_mean(shared_layer_two, axis=[1, 2], keepdims=True)
    scale = tf.sigmoid(attention)
    return input_feature * scale


def spatial_attention(input_feature):
    kernel_size = 7
    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    conv = tf.layers.conv2d(concat, filters=1, kernel_size=kernel_size, strides=1, padding='same',
                            activation=tf.nn.sigmoid)
    scale = input_feature * conv
    return scale


def CBAM(input_feature):
    channel_attention_feature = channel_attention(input_feature)
    spatial_attention_feature = spatial_attention(channel_attention_feature)
    return spatial_attention_feature


def alexnet_CBAM(input, is_training=True, reuse=False, scope='alexnet', flip=False):
    if flip == True:
        input = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input)
        input = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), input)

    with tf.variable_scope(scope, reuse=reuse):
        net = tf.contrib.layers.conv2d(input, num_outputs=96, kernel_size=11, stride=1, padding='VALID',
                                       activation_fn=tf.nn.relu, scope='conv1')
        net = tf.contrib.layers.max_pool2d(net, kernel_size=2, stride=2, padding='VALID', scope='pool1')
        net = tf.contrib.layers.conv2d(net, num_outputs=256, kernel_size=5, stride=1, padding='SAME',
                                       activation_fn=tf.nn.relu, scope='conv2')
        net = tf.contrib.layers.max_pool2d(net, kernel_size=1, stride=2, padding='VALID', scope='pool2')
        net = tf.contrib.layers.conv2d(net, num_outputs=384, kernel_size=3, stride=1, padding='SAME',
                                       activation_fn=tf.nn.relu, scope='conv3')
        net = tf.contrib.layers.conv2d(net, num_outputs=384, kernel_size=3, stride=1, padding='SAME',
                                       activation_fn=tf.nn.relu, scope='conv4')

        # 在第 4 层之后添加 CBAM 模块
        net = CBAM(net)

        net = tf.contrib.layers.conv2d(net, num_outputs=256, kernel_size=3, stride=1, padding='SAME',
                                       activation_fn=tf.nn.relu, scope='conv5')
        net = tf.contrib.layers.max_pool2d(net, kernel_size=1, stride=2, padding='VALID', scope='pool5')
        net = tf.contrib.layers.flatten(net, scope='flatten')
        net = tf.contrib.layers.fully_connected(net, num_outputs=4096, activation_fn=tf.nn.relu, scope='fc6')
        net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout6')
        net = tf.contrib.layers.fully_connected(net, num_outputs=4096, activation_fn=tf.nn.relu, scope='fc7')
        net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout7')
        output = tf.contrib.layers.fully_connected(net, num_outputs=2, activation_fn=None, scope='fc8')

    return output


def vgg_block(inputs, num_filters, num_convs, scope):
    with tf.variable_scope(scope):
        net = inputs
        for i in range(num_convs):
            net = slim.conv2d(net, num_filters, [3, 3], activation_fn=tf.nn.relu, scope='conv{}'.format(i + 1))
        net = slim.max_pool2d(net, [1, 1], scope='pool')
    return net


def vgg16(inputs, num_classes=2, is_training=True, reuse=False, scope='vgg_16', flip=False):
    if flip == True:
        inputs = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), inputs)
        inputs = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), inputs)
    with tf.variable_scope(scope, reuse=reuse):
        net = inputs
        net = vgg_block(net, 64, 2, 'block1')
        net = vgg_block(net, 128, 2, 'block2')
        net = vgg_block(net, 256, 3, 'block3')
        net = vgg_block(net, 512, 3, 'block4')
        net = vgg_block(net, 512, 3, 'block5')
        net = slim.flatten(net, scope='flatten')
        net = slim.fully_connected(net, 4096, activation_fn=tf.nn.relu, scope='fc6')
        net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout6')
        net = slim.fully_connected(net, 4096, activation_fn=tf.nn.relu, scope='fc7')
        net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout7')
        net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc8')
    return net


def depthwise_conv(input, kernel_size, stride=1, scope='depthwise_conv'):
    return slim.separable_convolution2d(input, num_outputs=None, kernel_size=kernel_size,
                                        depth_multiplier=1, stride=stride, padding='SAME',
                                        activation_fn=None, scope=scope)


def ghost_module(input, out_channels, ratio=2, stride=1, scope='ghost_module'):
    intermediate_channels = out_channels // ratio
    primary_conv = slim.conv2d(input, intermediate_channels, [1, 1], stride=1, padding='SAME',
                               scope=scope + '_primary_conv')
    depthwise = depthwise_conv(primary_conv, kernel_size=3, stride=stride, scope=scope + '_depthwise_conv')
    return tf.concat([primary_conv, depthwise], axis=-1)


def GhostNet(inputs, num_classes=2, is_training=True, reuse=False, scope='GhostNet', flip=False):
    if flip:
        inputs = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), inputs)
        inputs = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), inputs)
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.separable_convolution2d], activation_fn=tf.nn.relu6,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0)):
            net = slim.conv2d(inputs, 16, [3, 3], stride=2, padding='SAME', scope='conv1')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='maxpool1')

            # net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')

            # Ghost Module 1
            net = ghost_module(net, 16, scope='ghost_module1')
            net = depthwise_conv(net, kernel_size=3, stride=1, scope='depthwise_conv1')
            net = slim.conv2d(net, 48, [1, 1], stride=1, padding='SAME', scope='pointwise_conv1')
            # net = CBAM(net)

            # Ghost Module 2
            net = ghost_module(net, 48, scope='ghost_module2')
            net = depthwise_conv(net, kernel_size=3, stride=2, scope='depthwise_conv2')
            net = slim.conv2d(net, 96, [1, 1], stride=1, padding='SAME', scope='pointwise_conv2')
            # net = CBAM(net)

            # Ghost Module 3
            net = ghost_module(net, 96, scope='ghost_module3')
            net = depthwise_conv(net, kernel_size=3, stride=1, scope='depthwise_conv3')
            net = slim.conv2d(net, 192, [1, 1], stride=1, padding='SAME', scope='pointwise_conv3')

            # Ghost Module 4
            net = ghost_module(net, 192, scope='ghost_module4')
            net = depthwise_conv(net, kernel_size=3, stride=2, scope='depthwise_conv4')
            net = slim.conv2d(net, 384, [1, 1], stride=1, padding='SAME', scope='pointwise_conv4')

            # Ghost Module 5
            net = ghost_module(net, 384, scope='ghost_module5')
            net = depthwise_conv(net, kernel_size=3, stride=1, scope='depthwise_conv5')
            net = slim.conv2d(net, 768, [1, 1], stride=1, padding='SAME', scope='pointwise_conv5')

            # Ghost Module 6
            net = ghost_module(net, 768, scope='ghost_module6')
            net = slim.avg_pool2d(net, [1, 1], stride=1, padding='SAME', scope='avgpool')  # [7,7] SAME
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
            net = tf.squeeze(net, [1, 2], name='spatial_squeeze')
            return net


def GhostNet_V4(inputs, num_classes=2, is_training=True, reuse=False, scope='GhostNet', flip=False):
    if flip:
        inputs = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), inputs)
        inputs = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), inputs)
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.separable_convolution2d], activation_fn=tf.nn.relu6,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0)):
            net = slim.conv2d(inputs, 16, [3, 3], stride=2, padding='SAME', scope='conv1')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='maxpool1')

            # net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')

            # Ghost Module 1
            net = ghost_module(net, 16, scope='ghost_module1')
            net = depthwise_conv(net, kernel_size=3, stride=1, scope='depthwise_conv1')
            # net = slim.conv2d(net, 32, [3, 3], stride=1, padding='SAME', scope='pointwise_conv1')
            net = slim.conv2d(net, 32, [2, 2], scope='pointwise_conv1')
            # net = CBAM(net)

            # Ghost Module 2
            net = ghost_module(net, 32, scope='ghost_module2')
            net = depthwise_conv(net, kernel_size=3, stride=2, scope='depthwise_conv2')
            # net = slim.conv2d(net, 64, [3, 3], stride=1, padding='SAME', scope='pointwise_conv2')
            net = slim.conv2d(net, 64, [1, 1], scope='pointwise_conv2')
            # net = CBAM(net)

            # Ghost Module 3
            net = ghost_module(net, 64, scope='ghost_module3')
            net = depthwise_conv(net, kernel_size=3, stride=1, scope='depthwise_conv3')
            # net = slim.conv2d(net, 128, [3, 3], stride=1, padding='SAME', scope='pointwise_conv3')
            net = slim.conv2d(net, 128, [1, 1], scope='pointwise_conv3')

            # Ghost Module 4
            net = ghost_module(net, 128, scope='ghost_module4')
            net = depthwise_conv(net, kernel_size=3, stride=2, scope='depthwise_conv4')
            # net = slim.conv2d(net, 256, [3, 3], stride=1, padding='SAME', scope='pointwise_conv4')
            net = slim.conv2d(net, 256, [1, 1], scope='pointwise_conv4')

            # Ghost Module 5
            net = ghost_module(net, 256, scope='ghost_module5')
            net = depthwise_conv(net, kernel_size=3, stride=1, scope='depthwise_conv5')
            # net = slim.conv2d(net, 512, [3, 3], stride=1, padding='SAME', scope='pointwise_conv5')
            net = slim.conv2d(net, 512, [1, 1], scope='pointwise_conv5')

            # Ghost Module 6
            net = ghost_module(net, 512, scope='ghost_module6')
            net = slim.avg_pool2d(net, [1, 1], stride=1, padding='SAME', scope='avgpool')  # [7,7] SAME
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
            net = tf.squeeze(net, [1, 2], name='spatial_squeeze')
            return net


# 构建GhostNet主干网络
def Ghostnet_V3(inputs, num_classes=2, is_training=True, reuse=False, scope='GhostNet', flip=False):
    if flip:
        inputs = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), inputs)
        inputs = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), inputs)
    with tf.variable_scope(scope, reuse=reuse):
        net = slim.conv2d(inputs, 16, [3, 3], stride=2, scope='conv1')
        net = block.bottleneck(net, 16, [1, 1], stride=1, expansion=16, scope='bottleneck1')

        net = block.bottleneck(net, 24, [3, 3], stride=2, expansion=48, scope='bottleneck2')
        net = block.bottleneck(net, 24, [3, 3], stride=1, expansion=72, scope='bottleneck3')

        net = block.bottleneck(net, 40, [5, 5], stride=2, expansion=72, scope='bottleneck4')
        net = block.bottleneck(net, 40, [5, 5], stride=1, expansion=120, scope='bottleneck5')

        net = block.bottleneck(net, 80, [3, 3], stride=2, expansion=240, scope='bottleneck6')
        net = block.bottleneck(net, 80, [3, 3], stride=1, expansion=200, scope='bottleneck7')
        net = block.bottleneck(net, 80, [3, 3], stride=1, expansion=184, scope='bottleneck8')
        net = block.bottleneck(net, 80, [3, 3], stride=1, expansion=184, scope='bottleneck9')

        net = block.bottleneck(net, 112, [3, 3], stride=1, expansion=480, scope='bottleneck10')
        net = block.bottleneck(net, 112, [3, 3], stride=1, expansion=672, scope='bottleneck11')

        net = block.bottleneck(net, 160, [5, 5], stride=2, expansion=672, scope='bottleneck12')
        net = block.bottleneck(net, 160, [5, 5], stride=1, expansion=960, scope='bottleneck13')
        net = block.bottleneck(net, 160, [5, 5], stride=1, expansion=960, scope='bottleneck14')  # False True
        net = block.bottleneck(net, 160, [5, 5], stride=1, expansion=960, scope='bottleneck15')

        net = slim.conv2d(net, 960, [1, 1], scope='conv2')
        net = slim.avg_pool2d(net, net.shape[1:3], scope='avg_pool')
        net = slim.conv2d(net, 1280, [1, 1], scope='conv3')
        net = slim.flatten(net)
        # net = slim.fully_connected(net, 250, activation_fn=tf.nn.relu, scope='fc1')
        # net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
        logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc2')

    return logits


'''
input_shape: 代表输入特征图的尺寸
classes: 代表分类类别数量
ratio: Ghost模块中第一个1*1卷积下降通道数的倍数, 一般为2
'''


class KANNetwork(Model):
    def __init__(self, input_shape, num_classes):
        super(KANNetwork, self).__init__()
        self.kan_layers = [
            KANs.KANLinear(in_features=input, activation='softmax')
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.kan_layers:
            x = layer(x)
        return x
# def Ghostnet_V2(input_shape, classes=2, is_training=True, reuse=False, scope='GhostNet', flip=False, ratio=2):
#     if flip:
#         input_shape = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input_shape)
#         input_shape = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), input_shape)
#     # 构造输入层[224,224,3]
#     inputs = keras.Input(shape=input_shape)
#
#     # 标准卷积[224,224,3]==>[112,112,16]
#     x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=2, padding='same', use_bias=False)(inputs)
#     # 批标准化
#     x = layers.BatchNormalization()(x)
#     # relu激活
#     x = layers.Activation('relu')(x)
#
#     # [112,112,16]==>[112,112,16]
#     x = bneck(x, outputs_channel=16, kernel=(3, 3), strides=1, exp_channel=16, ratio=ratio, se=False)
#
#     # [112,112,16]==>[56,56,24]
#     x = bneck(x, outputs_channel=24, kernel=(3, 3), strides=2, exp_channel=48, ratio=ratio, se=False)
#
#     # [56,56,24]==>[56,56,24]
#     x = bneck(x, outputs_channel=24, kernel=(3, 3), strides=1, exp_channel=72, ratio=ratio, se=False)
#     # [56,56,24]==>[28,28,40]
#     x = bneck(x, outputs_channel=40, kernel=(5, 5), strides=2, exp_channel=72, ratio=ratio, se=True)
#
#     # [28,28,40]==>[28,28,40]
#     x = bneck(x, outputs_channel=40, kernel=(5, 5), strides=1, exp_channel=120, ratio=ratio, se=True)
#     # [28,28,40]==>[14,14,80]
#     x = bneck(x, outputs_channel=80, kernel=(3, 3), strides=2, exp_channel=240, ratio=ratio, se=False)
#
#     # [14,14,80]==>[14,14,80]
#     x = bneck(x, outputs_channel=80, kernel=(3, 3), strides=1, exp_channel=200, ratio=ratio, se=False)
#     # [14,14,80]==>[14,14,80]
#     x = bneck(x, outputs_channel=80, kernel=(3, 3), strides=1, exp_channel=184, ratio=ratio, se=False)
#     # [14,14,80]==>[14,14,80]
#     x = bneck(x, outputs_channel=80, kernel=(3, 3), strides=1, exp_channel=184, ratio=ratio, se=False)
#     # [14,14,80]==>[14,14,112]
#     x = bneck(x, outputs_channel=112, kernel=(3, 3), strides=1, exp_channel=480, ratio=ratio, se=True)
#     # [14,14,112]==>[14,14,112]
#     x = bneck(x, outputs_channel=112, kernel=(3, 3), strides=1, exp_channel=672, ratio=ratio, se=True)
#     # [14,14,112]==>[7,7,160]
#     x = bneck(x, outputs_channel=160, kernel=(5, 5), strides=2, exp_channel=672, ratio=ratio, se=True)
#
#     # [7,7,160]==>[7,7,160]
#     x = bneck(x, outputs_channel=160, kernel=(5, 5), strides=1, exp_channel=960, ratio=ratio, se=False)
#     # [7,7,160]==>[7,7,160]
#     x = bneck(x, outputs_channel=160, kernel=(5, 5), strides=1, exp_channel=960, ratio=ratio, se=True)
#     # [7,7,160]==>[7,7,160]
#     x = bneck(x, outputs_channel=160, kernel=(5, 5), strides=1, exp_channel=960, ratio=ratio, se=False)
#     # [7,7,160]==>[7,7,160]
#     x = bneck(x, outputs_channel=160, kernel=(5, 5), strides=1, exp_channel=960, ratio=ratio, se=True)
#
#     # 1*1卷积降维[7,7,160]==>[7,7,960]
#     x = layers.Conv2D(filters=960, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(x)
#     # 批标准化
#     x = layers.BatchNormalization()(x)
#     # relu激活
#     x = layers.Activation('relu')(x)
#
#     # [7,7,960]==>[None,960]
#     x = layers.GlobalAveragePooling2D()(x)
#     # 增加宽高维度[None,960]==>[1,1,960]
#     x = layers.Reshape(target_shape=(1, 1, x.shape[-1]))(x)
#
#     # 1*1卷积[1,1,960]==>[1,1,1280]
#     x = layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(x)
#     # 批标准化
#     x = layers.BatchNormalization()(x)
#     # relu激活
#     x = layers.Activation('relu')(x)
#
#     # 1*1卷积分类[1,1,1280]==>[1,1,classes]
#     x = layers.Conv2D(filters=classes, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(x)
#     # 删除宽高维度[b,1,1,classes]==>[b,classes]
#     outputs = tf.squeeze(x, axis=[1, 2])
#
#     # 构建模型
#     model = keras.Model(inputs, outputs)
#
#     # 返回模型
#     return model
