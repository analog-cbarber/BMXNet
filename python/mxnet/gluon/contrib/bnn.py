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

"""Gluon interface to Q* functions for binary neural networks"""
from mxnet.base import numeric_types
from mxnet.gluon.nn.conv_layers import _Conv

__all__ = ['QActivation', 'QDense', 'QConv1D', 'QConv2D', 'QConv3D']

from mxnet.gluon import HybridBlock


class QActivation(HybridBlock):
    r"""Quantized activation function.

    The following quantized/binarized activation are supported (operations are applied
    elementwisely to each scalar of the input tensor):

    - `1 bit`: using deteministic sign() function to generate binary activation
    - `2|4|8|16|32 bit`: using quantization function

    Attributes
    ----------
    act_bit : int (non-negative), optional, default=1
        Quantized activation function.
    backward_only : boolean, optional, default=False
        If set 'backward_only' to true, then the quantized activation processin forward pass will
        not be performed in this layer, the input data will be just copied to output. This
        setting is created for the combined use with QConvolution and QDense layers, since the
        quantized activation for input data will be done in the forward pass of those two layers.

    Inputs:
        - **data**: tensor of any shape.

    Outputs:
        - **out**: tensor with same shape as input.
    """

    def __init__(self, act_bit=1, backward_only=False, **kwargs):
        super(QActivation, self).__init__(**kwargs)
        self.act_bit = _check_act_bit(act_bit)
        self.backward_only = backward_only

    def hybrid_forward(self, F, x):
        return F.QActivation(x, act_bit=self.act_bit, backward_only=self.backward_only,
                             name='fwd')

    def __repr__(self):
        if self.backward_only:
            s = '{name}(bits={act_bit}, backward_only=True)'
        else:
            s = '{name}(bits={act_bit})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class QDense(HybridBlock):
    def __init__(self, units, act_bit=1, use_bias=True, flatten=True,
                 weight_initializer=None, bias_initializer='zeros',
                 in_units=0,
                 binarized_weights_only=False,
                 **kwargs):
        super(QDense, self).__init__(**kwargs)
        with self.name_scope():
            self.act_bit = _check_act_bit(act_bit)
            self.binarized_weights_only = binarized_weights_only
            self._flatten = flatten
            self._units = units
            self._in_units = self._check_in_units(in_units)

            # TODO deferred checking of in_units size
            #   Could use initializer wrapper, shape assertion operator (if one exists),
            #   or ???

            self.weight = self.params.get('weight', shape=(units, in_units),
                                          init=weight_initializer,
                                          allow_deferred_init=True)
            if use_bias:
                self.bias = self.params.get('bias', shape=(units,),
                                            init=bias_initializer,
                                            allow_deferred_init=True)
            else:
                self.bias = None


    def hybrid_forward(self, F, x, weight, bias=None):
        if self._flatten:
            x = F.Flatten(x)
        return F.QFullyConnected(x, weight, bias, no_bias=bias is None, num_hidden=self._units,
                                 act_bit=self.act_bit,
                                 binarized_weights_only=self.binarized_weights_only,
                                 name='fwd')

    def __repr__(self):
        s = '{name}({layout}, bits={act_bit})'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        act_bit=self.act_bit,
                        layout='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]))


    def _check_in_units(self, in_units):
        word_size = 64
        # TODO option to allow 32
        if in_units % word_size != 0:
            raise ValueError('in_units not a multiple of word size (%d)' % word_size)
        return in_units


class _QConv(_Conv):
    def __init__(self, channels, kernel_size, strides, padding, dilation,
                 groups, layout, in_channels=0, act_bit=1, use_bias=True,
                 weight_initializer=None, bias_initializer='zeros',
                 scaling_factor=0,
                 binarized_weights_only=False,
                 prefix=None, params=None):
        super(_QConv, self).__init__(channels, kernel_size, strides, padding, dilation, groups,
                                     layout,
                                     in_channels=in_channels,
                                     use_bias=use_bias,
                                     weight_initializer=weight_initializer,
                                     bias_initializer=bias_initializer,
                                     op_name='QConvolution',
                                     prefix=prefix,
                                     params=params)

        self._kwargs.update(dict(act_bit=act_bit,
                                 scaling_factor=scaling_factor,
                                 binarized_weights_only=binarized_weights_only))

    def _alias(self):
        return 'qconv'

    def __repr__(self):
        s = '{name}({mapping}, bits={act_bit}, kernel_size={kernel}, stride={stride}'
        len_kernel_size = len(self._kwargs['kernel'])
        if self._kwargs['pad'] != (0,) * len_kernel_size:
            s += ', padding={pad}'
        if self._kwargs['dilate'] != (1,) * len_kernel_size:
            s += ', dilation={dilate}'
        if hasattr(self, 'out_pad') and self.out_pad != (0,) * len_kernel_size:
            s += ', output_padding={out_pad}'.format(out_pad=self.out_pad)
        if self._kwargs['num_group'] != 1:
            s += ', groups={num_group}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        mapping='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]),
                        **self._kwargs)


class QConv1D(_QConv):
    def __init__(self, channels, kernel_size, strides=1, padding=0,
                 dilation=1, groups=1, layout='NCW',
                 act_bit=1, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0,
                 scaling_factor=0,
                 binarized_weights_only=False,
                 **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)
        assert len(kernel_size) == 1, "kernel_size must be a number or a list of 1 ints"
        super(QConv1D, self).__init__(channels, kernel_size, strides, padding, dilation, groups,
                                      layout, in_channels, act_bit, use_bias, weight_initializer,
                                      bias_initializer, scaling_factor, binarized_weights_only,
                                      **kwargs)



class QConv2D(_QConv):
    def __init__(self, channels, kernel_size, strides=(1,1), padding=(0,0),
                 dilation=(1,1), groups=1, layout='NCHW',
                 act_bit=1, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0,
                 scaling_factor=0,
                 binarized_weights_only=False,
                 **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        super(QConv2D, self).__init__(channels, kernel_size, strides, padding, dilation, groups,
                                      layout, in_channels, act_bit, use_bias, weight_initializer,
                                      bias_initializer, scaling_factor, binarized_weights_only,
                                      **kwargs)



class QConv3D(_QConv):
    def __init__(self, channels, kernel_size, strides=(1,1,1), padding=(0,0,0),
                 dilation=(1,1,1), groups=1, layout='NCDHW',
                 act_bit=1, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0,
                 scaling_factor=0,
                 binarized_weights_only=False,
                 **kwargs):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*3
        assert len(kernel_size) == 3, "kernel_size must be a number or a list of 3 ints"
        super(QConv3D, self).__init__(channels, kernel_size, strides, padding, dilation, groups,
                                      layout, in_channels, act_bit, use_bias, weight_initializer,
                                      bias_initializer, scaling_factor, binarized_weights_only,
                                      **kwargs)




def _check_act_bit(act_bit):
    """Verifies that act_bit keyword is a power of two."""
    if act_bit not in {1,2,4,8,16,32}:
        raise ValueError("Bad `act_bit` '%s' - not power of two no greater than 32" % act_bit)
    return act_bit