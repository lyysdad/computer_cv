import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import tensorflow.contrib.slim as slim

'''
ECA（Efficient Channel Attention）模块是一种轻量级的注意力机制，用于增强卷积神经网络中的特征表达。
ECA 模块通过全局平均池化获取每个通道的全局特征，并使用一维卷积操作（而不是全连接层）来捕获通道间的相互依赖关系，从而减少参数量和计算开销。
'''


def eca_block_V2(input_tensor, b=1, gamma=2):
    channel = input_tensor.shape[-1]
    kernel_size = int(abs((tf.log(tf.cast(channel, tf.float32)) / tf.log(2.0)) + b / gamma))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

    # Global Average Pooling
    avg_pool = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)

    # 1D Convolution to capture channel-wise dependencies
    conv1d = tf.layers.conv1d(avg_pool, filters=1, kernel_size=kernel_size, padding='same', use_bias=False)
    conv1d = tf.squeeze(conv1d, axis=-1)

    # Sigmoid activation to get the attention map
    attention = tf.nn.sigmoid(conv1d)

    # Reshape and multiply with input_tensor to apply attention
    attention = tf.reshape(attention, [-1, 1, 1, channel])
    output_tensor = input_tensor * attention

    return output_tensor


# SE注意力机制模块
def se_block(input_tensor, ratio=16):
    channel = int(input_tensor.shape[-1])
    se = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
    se = slim.conv2d(se, channel // ratio, [1, 1], activation_fn=tf.nn.relu, scope='fc1')
    se = slim.conv2d(se, channel, [1, 1], activation_fn=tf.nn.sigmoid, scope='fc2')
    return input_tensor * se


# ECA注意力机制模块
def eca_block(input_tensor, b=1, gamma=2):
    channel = int(input_tensor.shape[-1])
    # kernel_size = int(abs((tf.log(tf.cast(channel, tf.float32)) / tf.log(2.0)) + b / gamma))
    t = tf.cast(channel, tf.float32)
    kernel_size = tf.cast(tf.abs((tf.log(t) / tf.log(2.0)) + (b / gamma)), tf.int32)
    # kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
    kernel_size = kernel_size + (1 - kernel_size % 2)

    avg_pool = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
    # conv1d = tf.layers.conv1d(avg_pool, filters=1, kernel_size=kernel_size, padding='same', use_bias=False)
    conv1d = tf.layers.conv1d(avg_pool, filters=1, kernel_size=1, padding='same', use_bias=False)
    conv1d = tf.squeeze(conv1d, axis=-1)
    attention = tf.nn.sigmoid(conv1d)
    attention = tf.reshape(attention, [-1, 1, 1, channel])
    return input_tensor * attention


# Ghost模块
def ghost_module(inputs, output_channels, ratio=2):
    input_channels = int(inputs.shape[-1])
    hidden_channels = int(input_channels / ratio)

    primary_conv = slim.conv2d(inputs, hidden_channels, [1, 1], scope='primary_conv')
    cheap_operation = slim.separable_conv2d(primary_conv, None, [3, 3], depth_multiplier=ratio - 1,
                                            scope='cheap_operation')
    ghost_output = tf.concat([primary_conv, cheap_operation], axis=-1)
    ghost_output = slim.conv2d(ghost_output, output_channels, [1, 1], activation_fn=None, scope='ghost_output')

    return ghost_output


# Bottleneck模块
def bottleneck(inputs, output_channels, kernel_size, stride, expansion, se=False, eca=False, scope=None):
    with tf.variable_scope(scope, default_name='bottleneck'):
        shortcut = inputs
        inputs = ghost_module(inputs, expansion, ratio=2)
        inputs = slim.separable_conv2d(inputs, None, kernel_size, depth_multiplier=1, stride=stride,
                                       scope='depthwise_conv')
        inputs = slim.conv2d(inputs, output_channels, [1, 1], scope='project_conv')

        # if se:
        #     inputs = se_block(inputs)
        # if eca:
        #     inputs = eca_block(inputs)
        inputs = se_block(inputs)

        if stride == 1 and inputs.shape == shortcut.shape:
            inputs = inputs + shortcut

        return inputs


def coord_attention(input_feature, ratio=8):
    # 获取输入特征图的尺寸
    input_shape = input_feature.get_shape().as_list()
    # 获取特征图的高度和宽度
    H, W = input_shape[1], input_shape[2]

    # 计算空间坐标编码
    coord_h = tf.range(H, dtype=tf.float32) / (H - 1)
    coord_w = tf.range(W, dtype=tf.float32) / (W - 1)
    coord_h = 2 * coord_h - 1
    coord_w = 2 * coord_w - 1
    # coord_h = coord_h[None, :, None]
    # coord_w = coord_w[None, None, :]
    # coord_map = tf.concat([coord_h, coord_w], axis=0)
    coord_h = tf.expand_dims(coord_h, axis=0)
    coord_w = tf.expand_dims(coord_w, axis=-1)
    coord_map = tf.concat([coord_h, coord_w], axis=1)
    coord_map = tf.expand_dims(coord_map, axis=0)

    # 计算卷积核尺寸
    inter_channel = max(input_shape[3] // ratio, 1)
    # 定义坐标注意力机制的处理网络
    with tf.variable_scope('coord_att'):
        # 对特征图进行1x1卷积降维
        query = tf.layers.conv2d(input_feature, inter_channel, kernel_size=1)
        # 使用双线性插值对坐标特征图进行通道扩展
        key = tf.layers.conv2d_transpose(coord_map, inter_channel, kernel_size=3, strides=2, padding='same')
        key = tf.layers.conv2d(key, inter_channel, kernel_size=1)
        # 将query和key相乘并进行softmax归一化
        energy = tf.nn.softmax(query * key, axis=-1)
        # 对特征图进行1x1卷积升维
        value = tf.layers.conv2d(input_feature, input_shape[3], kernel_size=1)
        # 将value与归一化后的energy相乘得到加权后的特征图
        output_feature = value * energy
    return output_feature


class TripletAttention(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(TripletAttention, self).__init__()
        self.filters = filters
        self.channel_conv = tf.keras.layers.Conv2D(filters, kernel_size=1)
        self.spatial_conv = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
        self.cross_conv = tf.keras.layers.Conv2D(filters, kernel_size=1)

    def call(self, x):
        # Channel Attention
        avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        channel_attention = self.channel_conv(avg_pool + max_pool)

        # Spatial Attention
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_attention = self.spatial_conv(tf.concat([avg_pool, max_pool], axis=-1))

        # Cross Dimension Attention
        cross_attention = self.cross_conv(x)

        # Combining attentions
        output = x * channel_attention * spatial_attention * cross_attention

        return output


class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        # Channel Attention
        self.channel_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.channel_max_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.channel_shared_dense = tf.keras.layers.Dense(units=input_shape[-1] // self.reduction_ratio,
                                                          activation='relu', kernel_initializer='he_normal')
        self.channel_softmax_dense = tf.keras.layers.Dense(units=input_shape[-1], activation='sigmoid',
                                                           kernel_initializer='he_normal')

        # Spatial Attention
        self.spatial_conv = tf.keras.layers.Conv2D(filters=1, kernel_size=7, strides=1, padding='same',
                                                   activation='sigmoid')

    def call(self, x):
        # Channel Attention
        avg_pool = self.channel_avg_pool(x)
        max_pool = self.channel_max_pool(x)
        channel_shared = self.channel_shared_dense(tf.concat([avg_pool, max_pool], axis=1))
        channel_attention = self.channel_softmax_dense(channel_shared)
        channel_attention = tf.expand_dims(channel_attention, axis=1)
        channel_attention = tf.expand_dims(channel_attention, axis=1)

        channel_refined = tf.multiply(x, channel_attention)

        # Spatial Attention
        spatial_attention = self.spatial_conv(x)
        spatial_refined = tf.multiply(x, spatial_attention)

        # Combine Channel and Spatial Attention
        output = tf.add(channel_refined, spatial_refined)

        return output


class CoordAttention(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16, kernel_size=1):
        super(CoordAttention, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # 位置编码
        self.height = input_shape[1]
        self.width = input_shape[2]
        coord_x = tf.tile(tf.expand_dims(tf.range(self.width), axis=0), [self.height, 1])
        coord_x = tf.cast(coord_x, tf.float32)
        coord_y = tf.tile(tf.expand_dims(tf.range(self.height), axis=1), [1, self.width])
        # coord_x = tf.expand_dims(tf.expand_dims(tf.cast(coord_x, tf.float32) / self.width, axis=0), axis=-1)
        coord_x = tf.expand_dims(tf.expand_dims(coord_x / tf.cast(self.width, tf.float32), axis=0), axis=-1)
        # coord_y = tf.expand_dims(tf.expand_dims(tf.cast(coord_y, tf.float32) / self.height, axis=0), axis=-1)
        coord_y = tf.cast(coord_y, tf.float32)
        coord_y = tf.expand_dims(tf.expand_dims(coord_y / tf.cast(self.height, tf.float32), axis=0), axis=-1)
        self.coord = tf.concat([coord_x, coord_y], axis=-1)

        self.channels = int(self.channels)
        self.reduction_ratio = int(self.reduction_ratio)
        # 通道注意力
        self.conv_channel1 = tf.keras.layers.Conv2D(filters=self.channels // self.reduction_ratio,
                                                    kernel_size=self.kernel_size, padding='same', activation='relu')
        self.conv_channel2 = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel_size,
                                                    padding='same', activation='sigmoid')

        # 空间注意力
        self.conv_spatial1 = tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size, padding='same',
                                                    activation='relu')
        self.conv_spatial2 = tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size, padding='same',
                                                    activation='sigmoid')

    def call(self, x):
        # 通道注意力
        avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        channel_attention = self.conv_channel1(avg_pool)
        channel_attention = self.conv_channel2(channel_attention)

        # 空间注意力
        spatial_attention = self.conv_spatial1(self.coord)
        spatial_attention = self.conv_spatial2(spatial_attention)

        # 组合通道和空间注意力
        attention = tf.multiply(channel_attention, spatial_attention)

        # 加权特征
        weighted_features = tf.multiply(x, attention)

        return weighted_features


class MultiScaleDilatedAttention_V1(tf.keras.layers.Layer):
    def __init__(self, dilation_rates=[1, 2, 4], filters=64, kernel_size=3):
        super(MultiScaleDilatedAttention_V1, self).__init__()
        self.dilation_rates = dilation_rates
        self.filters = filters
        self.kernel_size = kernel_size

        # Convolutional layers for each dilation rate
        self.conv_layers = [
            tf.keras.layers.Conv2D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')
            for dilation_rate in self.dilation_rates]

        # Attention mechanism
        self.attention = tf.keras.layers.Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')

    def call(self, inputs):
        # Apply convolutional layers with different dilation rates
        conv_outputs = [conv_layer(inputs) for conv_layer in self.conv_layers]

        # Concatenate the outputs along the channel axis
        concatenated = tf.concat(conv_outputs, axis=-1)

        # Apply attention mechanism
        attention_output = self.attention(concatenated)

        # Weighted sum of convolutional outputs based on attention scores
        weighted_sum = tf.reduce_sum(attention_output * concatenated, axis=-1, keepdims=True)

        return weighted_sum


class DilateFormer(tf.keras.layers.Layer):
    def __init__(self, num_heads=4, head_size=64, dilation_rates=[1, 2, 4], filters=64):
        super(DilateFormer, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.dilation_rates = dilation_rates
        self.filters = filters

        self.mhsa_layers = [self.MultiHeadSelfAttention(num_heads=num_heads, head_size=head_size) for _ in range(len(dilation_rates))]
        self.dilated_convs = [tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', dilation_rate=dilation_rate, activation='relu') for dilation_rate in dilation_rates]

    class MultiHeadSelfAttention(tf.keras.layers.Layer):
        def __init__(self, num_heads, head_size):
            super(DilateFormer.MultiHeadSelfAttention, self).__init__()
            self.num_heads = num_heads
            self.head_size = head_size
            self.output_dim = num_heads * head_size

        def build(self, input_shape):
            self.query_dense = tf.keras.layers.Dense(self.output_dim)
            self.key_dense = tf.keras.layers.Dense(self.output_dim)
            self.value_dense = tf.keras.layers.Dense(self.output_dim)
            self.output_dense = tf.keras.layers.Dense(input_shape[-1])

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            query = self.query_dense(inputs)
            key = self.key_dense(inputs)
            value = self.value_dense(inputs)

            query = self.split_heads(query, batch_size)
            key = self.split_heads(key, batch_size)
            value = self.split_heads(value, batch_size)

            scaled_attention = self.scaled_dot_product_attention(query, key, value)

            concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.output_dim))
            output = self.output_dense(concat_attention)
            return output

        def split_heads(self, x, batch_size):
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_size))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        def scaled_dot_product_attention(self, query, key, value):
            matmul_qk = tf.matmul(query, key, transpose_b=True)
            dk = tf.cast(tf.shape(key)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.sqrt(dk)
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            output = tf.matmul(attention_weights, value)
            return output

    def call(self, inputs):
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, axis=1)

        mhsa_outputs = [mhsa_layer(inputs) for mhsa_layer in self.mhsa_layers]
        mhsa_outputs = [tf.expand_dims(mhsa_output, axis=1) for mhsa_output in mhsa_outputs]  # Ensure 4D for Conv2D

        conv_outputs = [dilated_conv(mhsa_output) for mhsa_output, dilated_conv in zip(mhsa_outputs, self.dilated_convs)]
        concatenated = tf.concat(conv_outputs, axis=-1)
        combined_output = tf.keras.layers.Conv2D(self.filters, kernel_size=1, padding='same', activation='relu')(concatenated)
        return combined_output

