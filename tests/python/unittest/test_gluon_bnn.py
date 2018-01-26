# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
"""Unit tests for mxnet.gluon.contrib.bnn module"""

from unittest import TestCase
from mxnet.gluon.contrib.bnn import *
import mxnet as mx
import numpy as np


class TestBNN(TestCase):
    def assertNDArrayAlmostEqual(self, expected_nd, actual_nd, places=None, delta=None):
        """Asserts that two NDArray instances have same shape and almost equal values."""
        self.assertEqual(expected_nd.shape, actual_nd.shape)
        expected_np = expected_nd.asnumpy()
        actual_np = actual_nd.asnumpy()

        iter = np.nditer([expected_np, actual_np], ['multi_index'])

        while not iter.finished:
            expected,actual = iter.value
            self.assertAlmostEqual(expected, actual,
                                   msg='%f != %f at index %s' % (expected,actual,
                                                                  iter.multi_index),
                                   places=places, delta=delta)
            iter.iternext()

    @staticmethod
    def aquantize(input, act_bit):
        """Quantize NDArray for activations using specified bit width"""
        if act_bit == 1:
            return (input >= 0).astype(np.float32) * 2 - 1

        clipped = mx.ndarray.clip(input, 0.0, 1.0)
        q = (1<<act_bit) - 1
        return mx.ndarray.rint(clipped * q) / q

    @staticmethod
    def wquantize(input, weight_bit):
        """Quantize NDArray for weights using specified bit width"""
        if weight_bit == 1:
            return (input >= 0).astype(np.float32) * 2 - 1
        elif weight_bit == 32:
            return input

        tanh_input = input.tanh()
        max_tanh_input = tanh_input.abs().max()
        squashed_input = tanh_input / (2 * max_tanh_input) + .5

        q = (1<<weight_bit) - 1
        quantized = 2 * (mx.ndarray.rint(squashed_input * q) / q) - 1.0
        return quantized

    def test_QActivation(self):
        """Unit test for QActivation block"""
        self.qact_case(mx.random.normal(shape=(1,5)))
        self.qact_case(mx.random.normal(shape=(3,5)), act_bit=2)
        self.qact_case(mx.random.normal(shape=(10,10)),
                  act_bit=2, backward_only=True)
        self.qact_case(mx.random.normal(shape=(3,3,3)),
                  act_bit=4)

        with self.assertRaises(ValueError):
            QActivation(act_bit=3)
        with self.assertRaises(ValueError):
            QActivation(act_bit=-1)
        with self.assertRaises(ValueError):
            QActivation(act_bit=33)

    def qact_case(self, input, **kwargs):
        """Run tests case for QActivation"""
        qact = QActivation(**kwargs)
        act_bit = kwargs.get('act_bit', 1)
        self.assertEqual(act_bit, qact.act_bit)
        backward_only = kwargs.get('backward_only', False)
        self.assertEqual(backward_only, qact.backward_only)

        if backward_only:
            expected_output = input
        else:
            expected_output = self.aquantize(input, act_bit)

        output = qact(input)
        self.assertNDArrayAlmostEqual(expected_output, output)

        # Try again in symbol mode
        qact.hybridize()
        output = qact(input)
        self.assertNDArrayAlmostEqual(expected_output, output)

        # TODO test backward

    def test_QDense(self):
        """Unit tests for QDense block"""
        self.qdense_case(mx.random.normal(shape=(1,64)), flatten=False)
        self.qdense_case(mx.random.normal(shape=(10,64)), flatten=False)
        self.qdense_case(mx.random.normal(shape=(10,128)))
        self.qdense_case(mx.random.normal(shape=(1,8,8)), act_bit=2, weight_bit=2)

        # try every combination
        for act_bit in [1,2,4,8,16,32]:
            for weight_bit in [1,2,4,8,16,32]:
                self.qdense_case(mx.random.normal(shape=(1,64)),
                                 act_bit=act_bit,
                                 weight_bit=weight_bit)

        # FIXME - int64 cannot currently be used as dtype of gluon Parameter
        # So this will not work until that is fixed or Q* functions use a
        # different representation for binary weights.
        #
        # self.qdense_case(mx.random.normal(shape=(1,64)), weight_dtype=np.int64, in_units=64)

        # TODO test bias

        # Error checking
        with self.assertRaisesRegexp(ValueError, '.*63 not a multiple.*'):
            self.qdense_case(None, in_units=63)
        with self.assertRaisesRegexp(ValueError, 'Unsupported weight_dtype.*'):
            self.qdense_case(None, weight_dtype=np.float64)
        with self.assertRaisesRegexp(ValueError, 'weight_dtype.*does not match wordsize'):
            self.qdense_case(None, weight_dtype=np.int32, wordsize=64)
        with self.assertRaisesRegexp(ValueError, 'weight_dtype.*does not match wordsize'):
            self.qdense_case(None, weight_dtype=np.int64, wordsize=32)
        with self.assertRaisesRegexp(ValueError, '.*only supported when act_bit.*'):
            self.qdense_case(None, weight_dtype=np.int64, act_bit=2)
        with self.assertRaisesRegexp(NotImplementedError, 'bias not supported with binary '
                                                          'weights.*'):
            self.qdense_case(None, use_bias=True)

    def qdense_case(self, input, units=4, **kwargs):
        """Run test case for QDense block"""
        qdense = QDense(units, **kwargs)

        in_units = kwargs.get('in_units', 0)
        flatten = kwargs.get('flatten', True)
        act_bit = kwargs.get('act_bit', 1)
        prepend_act = kwargs.get('prepend_act', True)
        weight_bit = kwargs.get('weight_bit', 1)
        weight_initializer = kwargs.get('weight_initializer', None)
        use_bias = kwargs.get('use_bias', False)
        bias_initializer = kwargs.get('bias_initializer', 'zeros')
        weight_dtype = kwargs.get('weight_dtype', np.float32)
        wordsize = kwargs.get('wordsize', 64)

        self.assertEqual(units, qdense.units)
        self.assertEqual(in_units, qdense.in_units)
        self.assertEqual(act_bit, qdense.act_bit)
        self.assertEqual(weight_bit, qdense.weight_bit)
        self.assertEqual(prepend_act, qdense.prepend_act)

        self.assertEqual(weight_dtype, qdense.weight.dtype)

        # Run forward
        qdense.collect_params().initialize()

        # Run once to init params, which we will then set manually
        W = mx.random.normal(shape=(units, np.prod(input.shape[1:])))
        qdense(input)
        qdense.weight.set_data(W)
        output = qdense(input)

        issue30Fixed = False
        if issue30Fixed:
            # issue #30 - weights should not be modified when running forward!
            self.assertNDArrayAlmostEqual(W, qdense.weight.data())
        else:
            qdense.weight.set_data(W)

        qinput = self.aquantize(mx.ndarray.flatten(input), act_bit)
        if qdense.binarized_weights:
            qweight = None # TODO convert from binary format weights
        else:
            qweight = self.wquantize(W, weight_bit)

        if qweight is None:
            expected_output = output
        else:
            qbias = None
            if use_bias:
                qbias = qdense.bias.data()
            expected_output = mx.ndarray.FullyConnected(qinput, qweight, qbias,
                                                        num_hidden=units,
                                                        no_bias=qbias is None,
                                                        flatten=flatten)
            if act_bit == 1 and weight_bit == 1:
                n_inputs = qinput.shape[1]
                expected_output = (expected_output + n_inputs) / 2.0

        self.assertNDArrayAlmostEqual(expected_output, output, places=4)

        # try again in symbol mode
        qdense.hybridize()
        if not issue30Fixed:
            qdense.weight.set_data(W)
        output = qdense(input)
        self.assertNDArrayAlmostEqual(expected_output, output, places=4)

    def test_QConv(self):
        """Unit tests for QConv* blocks"""
        for act_bit in [1,2,4,8,16,32]:
            for weight_bit in [1,2,4,8,16,32]:
                for dim in [2]: # TODO support 1 and 3
                    input_shape=(1,64) + (8,)*dim
                    self.qconv_case(mx.random.normal(shape=input_shape),
                                    act_bit=act_bit, weight_bit=weight_bit)

        self.qconv_case(mx.random.normal(shape=(1,64,8,8)), act_bit=1, weight_bit=1)
        self.qconv_case(mx.random.normal(shape=(1,64,8,8)), act_bit=2, weight_bit=2)

        # Error checking
        input = mx.random.normal(shape=(1,64,8,8))
        with self.assertRaisesRegexp(ValueError, '.*63 not a multiple.*'):
            self.qconv_case(input, in_channels=63)
        with self.assertRaisesRegexp(ValueError, 'Unsupported weight_dtype.*'):
            self.qconv_case(input, weight_dtype=np.float64)
        with self.assertRaisesRegexp(ValueError, 'weight_dtype.*does not match wordsize'):
            self.qconv_case(input, weight_dtype=np.int32, wordsize=64)
        with self.assertRaisesRegexp(ValueError, 'weight_dtype.*does not match wordsize'):
            self.qconv_case(input, weight_dtype=np.int64, wordsize=32)
        with self.assertRaisesRegexp(ValueError, '.*only supported when act_bit.*'):
            self.qconv_case(input, weight_dtype=np.int64, act_bit=2)
        with self.assertRaisesRegexp(NotImplementedError, 'bias not supported with binary '
                                                          'weights.*'):
            self.qconv_case(input, use_bias=True)


    def qconv_case(self, input, channels=4, kernel=3, **kwargs):
        dim = len(input.shape) - 2 # subtract batch and channel dimensions

        if dim == 1:
            qconv = QConv1D(channels, kernel, **kwargs)
            default_layout = 'NCW'
        elif dim == 2:
            qconv = QConv2D(channels, kernel, **kwargs)
            default_layout = 'NCHW'
        elif dim == 3:
            qconv = QConv3D(channels, kernel, **kwargs)
            default_layout = 'NCDHW'
        else:
            self.fail('too many dimensions in input')

        def tuplify(x):
            return (x,) * dim if isinstance(x, int) else x

        kernel = tuplify(kernel)
        strides = tuplify(kwargs.get('strides', 1))
        padding = tuplify(kwargs.get('padding', 0))
        dilation = tuplify(kwargs.get('dilation', 1))
        groups = kwargs.get('groups', 1)
        in_channels = kwargs.get('in_channels', 0)
        act_bit = kwargs.get('act_bit', 1)
        prepend_act = kwargs.get('prepend_act', True)
        weight_bit = kwargs.get('weight_bit', 1)
        weight_initializer = kwargs.get('weight_initializer', None)
        weight_dtype = kwargs.get('weight_dtype', np.float32)
        use_bias = kwargs.get('use_bias', False)
        bias_initializer = kwargs.get('bias_initializer', 'zeros')
        scaling_factor = kwargs.get('scaling_factor', False)
        wordsize = kwargs.get('wordsize', 64)
        cudnn_tune = kwargs.get('cudnn_tune', None)
        cudnn_off = kwargs.get('cudnn_off', False)


        layout = kwargs.get('layout', default_layout)

        self.assertEqual(act_bit, qconv.act_bit)
        self.assertEqual(weight_bit, qconv.weight_bit)
        self.assertEqual(prepend_act, qconv.prepend_act)

        self.assertEqual(np.dtype(weight_dtype).type, qconv.weight.dtype)

        wshape = qconv.weight.shape
        self.assertEqual(dim+2, len(wshape))
        self.assertEqual((channels, in_channels) + kernel, wshape)

        qconv.collect_params().initialize()

        # Run once to init params, which we will then set manually
        # This is to work around BMXNet issue #30.
        qconv(input)
        W = mx.random.normal(shape=(qconv.weight.shape))
        qconv.weight.set_data(W)

        qconv(input)
        self.assertNDArrayAlmostEqual(W, qconv.weight.data())

        issue38Fixed = False

        with mx.autograd.train_mode(): # HACK around issue #38
            output = qconv(input)
        if issue38Fixed:
            self.assertNDArrayAlmostEqual(W, qconv.weight.data())

        qinput = self.aquantize(input, act_bit)
        if qconv.binarized_weights:
            qweight = None # TODO convert from binary format weights
        else:
            qweight = self.wquantize(W, weight_bit)

        if qweight is None:
            expected_output = output
        else:
            qbias = None
            if use_bias:
                qbias = qconv.bias.data()
            expected_output = mx.ndarray.Convolution(qinput, qweight, qbias,
                                                     kernel=kernel,
                                                     stride=strides,
                                                     dilate=dilation,
                                                     pad=padding,
                                                     num_filter=channels,
                                                     num_group=groups,
                                                     no_bias=qbias is None,
                                                     cudnn_tune=cudnn_tune,
                                                     cudnn_off=cudnn_off,
                                                     layout=layout)
            if act_bit == 1 and weight_bit == 1:
                n_inputs = np.prod(W.shape[1:])
                expected_output = (expected_output + n_inputs) / 2.0

        self.assertNDArrayAlmostEqual(expected_output, output, places=3)

        # try again in symbol mode
        qconv.hybridize()
        if not issue38Fixed:
            qconv.weight.set_data(W)
        with mx.autograd.train_mode(): # HACK around bug
            output = qconv(input)
        self.assertNDArrayAlmostEqual(expected_output, output, places=3)
