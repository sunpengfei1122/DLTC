# -*- coding: utf-8 -*-
# File: fc.py


import numpy as np
from ..compat import tfv1 as tf  # this should be avoided first in model code

from ..tfutils.common import get_tf_version_tuple
from .common import VariableHolder, layer_register
from .tflayer import convert_to_tflayer_args, rename_get_variable

__all__ = ['sFullyConnected']


import tensorflow as tf

class sFullyConnected_layer(tf.keras.layers.Layer):
    def __init__(self, units, use_bias= False, name= None, activation=None, **kwargs):
        super(sFullyConnected_layer, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        #self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # 初始化对角线上的权重
        input_shape = tf.TensorShape(input_shape)
        #input_dim = int(input_shape[-1])
        self.kernel  = self.add_weight(
            name='kernel',
            shape=(self.units,),
            initializer= tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal'),
            trainable=True,
         )

            # 初始化偏置
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )
        self.built = True


    def call(self, inputs):
        # 将权重转换为对角矩阵
        #diag_weights_matrix = tf.linalg.diag(self.diag_weights)

        # 计算输出
        output = inputs * self.diag_weights
        if self.use_bias:
              output = output + self.bias

        # 应用激活函数（如果有的话）
        #if self.activation is not Noneunits:
        #    output = self.activation(output)

        return output

def sFullyConnected(name, 
        inputs,
        units,
        activation=None,
        use_bias=False,
        input_shape = None, 
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            input_shape = inputs.get_shape().as_list()
            output_channel = input_shape[-1]  
            W = tf.get_variable(
            'W', [output_channel], initializer=tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal'))

            ret = W * inputs  

            if use_bias:
                b = tf.get_variable('b', [output_channel], initializer=tf.zeros_initializer())
                ret = tf.nn.bias_add(ret, b, data_format='NHWC') 

            ret = tf.identity(ret, name='output')

            ret.variables = VariableHolder(W=W)
            if use_bias:
                ret.variables.b = b
    return ret

'''
def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['units'],
    name_mapping={'out_dim': 'units'})
def sFullyConnected(
        inputs,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None):
    """
    A wrapper around `tf.layers.Dense`.
    One difference to maintain backward-compatibility:
    Default weight initializer is variance_scaling_initializer(2.0).

    Variable Names:

    * ``W``: weights of shape [in_dim, out_dim]
    * ``b``: bias
    """
    if kernel_initializer is None:
        if get_tf_version_tuple() <= (1, 12):
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)  # deprecated
        else:
            kernel_initializer = tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal')

    #inputs = batch_flatten(inputs)
    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            _reuse=tf.get_variable_scope().reuse)
        ret = layer.apply(inputs, scope=tf.get_variable_scope())
        ret = tf.identity(ret, name='output')

    ret.variables = VariableHolder(W=layer.kernel)
    if use_bias:
        ret.variables.b = layer.bias
    return ret
'''
