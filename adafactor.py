#! -*- coding: utf-8 -*-

import os, sys
from distutils.util import strtobool
import numpy as np
import tensorflow as tf

# 判断是tf.keras还是纯keras的标记
is_tf_keras = strtobool(os.environ.get('TF_KERAS', '0'))

if is_tf_keras:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    sys.modules['keras'] = keras
else:
    import keras
    import keras.backend as K


class AdaFactorBase(keras.optimizers.Optimizer):
    """AdaFactor优化器（基类）
    论文链接：https://arxiv.org/abs/1804.04235
    参考实现：https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/optimize.py
    """
    def __init__(
            self,
            learning_rate=1e-3,  # 可以为None
            beta1=0.0,
            beta2=None,
            epsilon1=1e-30,
            epsilon2=1e-3,
            multiply_by_parameter_scale=True,
            clipping_threshold=1.0,
            min_dim_size_to_factor=128,
            **kwargs):
        super(AdaFactorBase, self).__init__(**kwargs)
        self._learning_rate = learning_rate
        self.beta1 = beta1
        self._beta2 = beta2
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.multiply_by_parameter_scale = multiply_by_parameter_scale
        self.clipping_threshold = clipping_threshold
        self.min_dim_size_to_factor = min_dim_size_to_factor

    @property
    def learning_rate(self):
        if self._learning_rate is None:
            iterations = K.cast(self.iterations + 1, K.floatx())
            learning_rate = K.minimum(1.0 / K.sqrt(iterations), 0.01)
            if self.multiply_by_parameter_scale:
                return learning_rate
            else:
                return learning_rate * 0.05
        else:
            if not hasattr(self, '__learning_rate'):
                with K.name_scope(self.__class__.__name__):
                    self.__learning_rate = K.variable(self._learning_rate,
                                                      name='learning_rate')
            return self.__learning_rate

    @property
    def beta2(self):
        if self._beta2 is None:
            iterations = K.cast(self.iterations + 1, K.floatx())
            return 1.0 - K.pow(iterations, -0.8)
        else:
            return self._beta2

    def factored_shape(self, shape):
        if len(shape) < 2:
            return None
        shape = np.array(shape)
        indices = shape.argpartition(-2)
        if shape[indices[-2]] < self.min_dim_size_to_factor:
            return None
        shape1, shape2 = np.array(shape), np.array(shape)
        shape1[indices[-1]] = 1
        shape2[indices[-2]] = 1
        return shape1, indices[-1], shape2, indices[-2]

    def get_config(self):
        config = {
            'learning_rate': self._learning_rate,
            'beta1': self.beta1,
            'beta2': self._beta2,
            'epsilon1': self.epsilon1,
            'epsilon2': self.epsilon2,
            'multiply_by_parameter_scale': self.multiply_by_parameter_scale,
            'clipping_threshold': self.clipping_threshold,
            'min_dim_size_to_factor': self.min_dim_size_to_factor,
        }
        base_config = super(AdaFactorBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdaFactorV1(AdaFactorBase):
    """AdaFactor优化器（纯Keras版）
    论文链接：https://arxiv.org/abs/1804.04235
    参考实现：https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/optimize.py
    """
    def __init__(self, *args, **kwargs):
        super(AdaFactorV1, self).__init__(*args, **kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')

    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.weights = [self.iterations]
        lr = self.learning_rate

        for i, (p, g) in enumerate(zip(params, grads)):
            g2 = K.square(g) + self.epsilon1
            shape, dtype = K.int_shape(p), K.dtype(p)
            factored_shape = self.factored_shape(shape)
            if factored_shape is None:
                # 定义参数
                v = K.zeros(shape, dtype=dtype, name='v_' + str(i))
                self.weights.append(v)
                # 定义更新
                v_t = self.beta2 * v + (1.0 - self.beta2) * g2
                self.updates.append(K.update(v, v_t))
            else:
                # 定义参数
                shape1, axis1, shape2, axis2 = factored_shape
                vr = K.zeros(shape1, dtype=dtype, name='vr_' + str(i))
                vc = K.zeros(shape2, dtype=dtype, name='vc_' + str(i))
                self.weights.extend([vr, vc])
                # 定义更新
                vr_t = self.beta2 * vr + K.mean(g2, axis=axis1, keepdims=True)
                vc_t = self.beta2 * vc + K.mean(g2, axis=axis2, keepdims=True)
                self.updates.extend([K.update(vr, vr_t), K.update(vc, vc_t)])
                # 合成矩阵
                v_t = vr_t * vc_t / K.mean(vr_t, axis=axis2, keepdims=True)
            # 增量主体
            u = g / K.sqrt(v_t)
            # 增量裁剪
            if self.clipping_threshold is not None:
                u_rms = K.mean(K.sum(K.square(u)))
                d = self.clipping_threshold
                u = u / K.maximum(1.0, u_rms / d)
            # 增量滑动
            if self.beta1 > 0.0:
                # 定义参数
                m = K.zeros(shape, dtype=dtype, name='m_' + str(i))
                self.weights.append(m)
                # 定义更新
                m_t = self.beta1 * m + (1.0 - self.beta1) * u
                self.updates.append(K.update(m, m_t))
                u = m_t
            # 增量调整
            if self.multiply_by_parameter_scale:
                u = u * K.maximum(K.mean(K.sum(K.square(p))), self.epsilon2)
            # 更新参数
            self.updates.append(K.update(p, p - lr * u))

        return self.updates


class AdaFactorV2(AdaFactorBase):
    """AdaFactor优化器（tf.keras版）
    论文链接：https://arxiv.org/abs/1804.04235
    参考实现：https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/optimize.py
    """
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'AdaFactor'
        super(AdaFactorV2, self).__init__(*args, **kwargs)

    def _create_slots(self, var_list):
        for var in var_list:
            if self.beta1 > 0.0:
                self.add_slot(var, 'm')
            shape = K.int_shape(var)
            factored_shape = self.factored_shape(shape)
            if factored_shape is None:
                self.add_slot(var, 'v')
            else:
                shape1, axis1, shape2, axis2 = factored_shape
                value1, value2 = np.zeros(shape1), np.zeros(shape2)
                self.add_slot(var, 'vr', value1)
                self.add_slot(var, 'vc', value2)

    def _resource_apply_dense(self, grad, var):
        lr = self.learning_rate
        g2 = K.square(grad) + self.epsilon1
        shape = K.int_shape(var)
        factored_shape = self.factored_shape(shape)
        if factored_shape is None:
            v = self.get_slot(var, 'v')
            # 定义更新
            v_t = self.beta2 * v + (1.0 - self.beta2) * g2
            v_t = K.update(v, v_t)
        else:
            shape1, axis1, shape2, axis2 = factored_shape
            vr = self.get_slot(var, 'vr')
            vc = self.get_slot(var, 'vc')
            # 定义更新
            vr_t = self.beta2 * vr + K.mean(g2, axis=axis1, keepdims=True)
            vc_t = self.beta2 * vc + K.mean(g2, axis=axis2, keepdims=True)
            vr_t, vc_t = K.update(vr, vr_t), K.update(vc, vc_t)
            # 合成矩阵
            v_t = vr_t * vc_t / K.mean(vr_t, axis=axis2, keepdims=True)
        # 增量主体
        u = grad / K.sqrt(v_t)
        # 增量裁剪
        if self.clipping_threshold is not None:
            u_rms = K.mean(K.sum(K.square(u)))
            d = self.clipping_threshold
            u = u / K.maximum(1.0, u_rms / d)
        # 增量滑动
        if self.beta1 > 0.0:
            m = self.get_slot(var, 'm')
            # 定义更新
            m_t = self.beta1 * m + (1.0 - self.beta1) * u
            u = K.update(m, m_t)
        # 增量调整
        if self.multiply_by_parameter_scale:
            u = u * K.maximum(K.mean(K.sum(K.square(var))), self.epsilon2)
        # 更新参数
        return K.update(var, var - lr * u)

    def _resource_apply_sparse(self, grad, var, indices):
        grad = tf.IndexedSlices(grad, indices, K.shape(var))
        grad = tf.convert_to_tensor(grad)
        return self._resource_apply_dense(grad, var)


if is_tf_keras:
    AdaFactor = AdaFactorV2
else:
    AdaFactor = AdaFactorV1
