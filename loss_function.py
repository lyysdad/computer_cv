import tensorflow as tf
from tensorflow.python.ops import array_ops
"""
Focal Loss损失函数的实现
:param y_true: 真实标签
:param y_pred: 预测标签
:param gamma: 调节因子 (default is 2.0)
:param alpha: 平衡因子 (default is 0.25)
:return: Focal Loss
"""


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    prob = tf.nn.softmax(y_pred)
    focal_weights = alpha * tf.pow(1 - prob, gamma)
    focal_weights = tf.reshape(focal_weights, [-1, 1])
    focal_loss = focal_weights * cross_entropy

    # 返回损失的均值
    return focal_loss


def focal_loss_V2(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.

        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.

    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    # return tf.reduce_sum(per_entry_cross_ent)
    return per_entry_cross_ent

def cross_entropy_tf(logits, labels, class_number):
    """TF交叉熵损失函数"""
    labels = tf.one_hot(labels, class_number)
    ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return ce_loss


# class Poly1CrossEntropyLoss(tf.keras.losses.Loss):
#
#     def __init__(self, num_classes, epsilon=1.0, reduction="sum", weight=None, name="poly_cross_entropy", **kwargs):
#         super(Poly1CrossEntropyLoss, self).__init__(name=name, **kwargs)
#         self.num_classes = num_classes
#         self.epsilon = epsilon
#         self.weight = weight
#
#     def call(self, labels, logits):
#
#         labels_onehot = tf.one_hot(labels, depth=self.num_classes, dtype=logits.dtype)
#         pt = tf.reduce_sum(labels_onehot * tf.nn.softmax(logits, axis=-1), axis=-1)
#         ce_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)  # , from_logits=True
#         poly1 = ce_loss + self.epsilon * (1 - pt)
#
#         if self.reduction == "mean":
#             poly1 = tf.reduce_mean(poly1)
#         elif self.reduction == "sum":
#             poly1 = tf.reduce_sum(poly1)
#
#         return poly1
#
#     def get_config(self):
#         config = {
#             'weight': self.weight,
#             'epsilon': self.epsilon
#         }
#         base_config = super().get_config()
#         return {**base_config, **config}


    # def poly1_cross_entropy_tf(y_true, y_pred, class_number=2, epsilon=1.0):
#     """poly_loss针对交叉熵损失函数优化，使用增加第一个多项式系数"""
#     labels = tf.cast(y_true, tf.int64)
#     # predicted = tf.cast(y_pred, tf.int64)
#     labels = tf.one_hot(labels, class_number)
#     # predicted = tf.one_hot(predicted, class_number)
#     y_pred_softmax = tf.nn.softmax(y_pred)  # 对 y_pred 进行 softmax 操作
#     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_pred_softmax)
#     poly1 = tf.reduce_sum(labels * tf.nn.softmax(y_pred), axis=-1)
#     poly1_loss = cross_entropy + epsilon * (1 - poly1)
#     return poly1_loss

# def poly1_cross_entropy_tf(y_true, y_pred, class_number=2, epsilon=1.0):
#     """poly_loss针对交叉熵损失函数优化，使用增加第一个多项式系数"""
#     y_true = tf.cast(y_true, tf.int64)
#     labels = tf.one_hot(y_true, class_number)
#     # y_pred_softmax = tf.nn.softmax(y_pred)  # 对 y_pred 进行 softmax 操作
#     # y_pred_softmax = tf.one_hot(y_pred_softmax, class_number)  # 将 y_pred 转换为 one-hot 编码
#     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_pred)
#     labels = tf.reduce_sum(labels, axis=-1)
#     poly1 = tf.reduce_sum(labels * y_pred, axis=-1)
#     poly1_loss = cross_entropy + epsilon * (1 - poly1)
#     return poly1_loss


# def poly1_cross_entropy_tf(y_true, y_pred, class_number=2, epsilon=1.0):
#     """poly_loss针对交叉熵损失函数优化，使用增加第一个多项式系数"""
#     labels = tf.one_hot(y_true, class_number)
#     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_pred)
#     poly1_loss = cross_entropy + epsilon
#     return poly1_loss

'''
PolyCross是一种用于图像识别的注意力机制，它是由Google Brain团队提出的一种新型注意力机制。
在TensorFlow中，我们可以使用自定义的TensorFlow操作来实现PolyCross。
'''
def poly_cross_attention(query, key, value, output_dim, key_dim, alpha=0.2):
    # 计算查询和键的交叉相关性
    score = tf.matmul(query, key, transpose_b=True)
    # 添加alphaDropout以提高鲁棒性
    dropout_score = alpha * score
    # 应用softmax函数得到注意力权重
    weights = tf.nn.softmax(dropout_score)
    # 进行注意力加权并获取输出
    result = tf.matmul(weights, value)
    # 使用全连接层进行投影，得到输出的维度
    return tf.layers.dense(result, output_dim, use_bias=False, name='context_output')


def focal_loss_tf(logits, labels, class_number=2, alpha=0.25, gamma=2.0, epsilon=1.e-7):
    """TF focal_loss函数"""
    alpha = tf.constant(alpha, dtype=tf.float32)
    y_true = tf.one_hot(0, class_number)
    alpha = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
    labels = tf.cast(labels, dtype=tf.int32)
    print(logits.shape)
    logits = tf.cast(logits, tf.float32)
    softmax = tf.reshape(tf.nn.softmax(logits), [-1])
    labels_shift = tf.range(0, logits.shape[0]) * logits.shape[1] + labels
    prob = tf.gather(softmax, labels_shift)
    prob = tf.clip_by_value(prob, epsilon, 1. - epsilon)
    alpha_choice = tf.gather(alpha, labels)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    weight = tf.multiply(alpha_choice, weight)
    fc_loss = -tf.multiply(weight, tf.log(prob))
    return fc_loss


def poly1_focal_loss_tf(logits, labels, class_number=2, alpha=0.25, gamma=2.0, epsilon=1.0):
    # fc_loss = focal_loss_tf(logits, labels, class_number, alpha, gamma)
    fc_loss = focal_loss(logits, labels, alpha, gamma)
    p = tf.nn.softmax(logits)
    labels = tf.cast(labels, tf.int64)
    labels = tf.one_hot(labels, class_number)
    labels_reduced = tf.reduce_sum(labels, axis=-1)
    poly1 = labels_reduced * p + (1 - labels_reduced) * (1 - p)
    poly1_loss = fc_loss + tf.reduce_mean(epsilon * tf.pow(1 - poly1, 2 + 1), axis=-1)
    return poly1_loss

def poly1_cross_entropy_tf(y_true, y_pred, class_number=2, epsilon=1.0):
    """poly_loss针对交叉熵损失函数优化，使用增加第一个多项式系数"""
    y_true = tf.cast(y_true, tf.int64)
    y_pred = tf.nn.softmax(y_pred)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    labels = tf.one_hot(y_true, class_number)
    labels = tf.reduce_sum(labels, axis=-1)
    poly1 = tf.reduce_sum(labels * y_pred, axis=-1)
    poly1_loss = cross_entropy + epsilon * (1 - poly1)
    return poly1_loss
