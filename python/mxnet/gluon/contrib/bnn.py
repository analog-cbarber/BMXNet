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

"""Gluon interface to Q* functions for quantized/binarized neural networks"""
import numpy
import numpy as np

from mxnet.base import numeric_types
from mxnet.gluon.nn.conv_layers import _Conv

__all__ = ['QActivation', 'QDense', 'QConv2D']

from mxnet.gluon import HybridBlock


class QActivation(HybridBlock):
    r"""Quantized activation function.

    The following quantized/binarized activation are supported (operations are applied
    elementwisely to each scalar of the input tensor):

    - `1 bit`: using deterministic sign() function to generate binary activation, i.e.
      outputs will be -1 or 1

    - `2|4|8|16|32 bit`: quantizes input clipped to range [0,1] uniformly to 2^act_bit
      values spaced uniformly from 0 to 1.

    Note that the return type is still a 32-bit float.

    Inputs:
        - **data**: tensor of any shape with dtype float32.

    Outputs:
        - **out**: tensor with same shape as input with dtype float32.
    """

    def __init__(self, act_bit=1, backward_only=False, **kwargs):
        """Construct QActivation block

        Parameters
        ----------
        act_bit : int (non-negative), optional, default=1
            Number of quantization bits used for each activation.
            Must be power of two no greater than 32.

        backward_only : boolean, optional, default=False
            If set 'backward_only' to true, then the quantized activation processing forward pass will
            not be performed in this layer, the input data will be just copied to output. This
            setting is created for the combined use with QConvolution and QDense layers, since the
            quantized activation for input data will be done in the forward pass of those two layers.

        **kwargs: see `mxnet.gluon.Block`
        """
        super(QActivation, self).__init__(**kwargs)
        self._act_bit = _check_bit_arg('act_bit', act_bit)
        self._backward_only = backward_only

    @property
    def act_bit(self):
        """Number of bits in quantized representation."""
        return self._act_bit

    @property
    def backward_only(self):
        """If true, block will simply pass input through unchanged running forward."""
        return self._backward_only

    def hybrid_forward(self, F, x):
        return F.QActivation(x, act_bit=self._act_bit, backward_only=self._backward_only,
                             name='fwd')

    def __repr__(self):
        s = '{name}(act_bit={_act_bit}, backward_only={_backward_only})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class QDense(HybridBlock):
    r"""Fully connected NN layer that quantizes weights and inputs.

    This block implements a quantized version of matrix-vector multiplication.
    The quantization serves as the activation function for the layer.

    This will quantize the input activations using the act_bit parameter
    to determine the number of quantization bins. If act_bit is one,
    inputs are quantized to -1/+1, otherwise they are quantized to
    2^act_bit bins spanning the range [0,1].

    If weight_bit is one, the weights are quantized to -1/1, otherwise
    the weights are first squashed to interval [0,1] by applying:

      (1 + tanh(w)) / 2 max(|tanh(w)|)

    and then quantized to 2^weight_bit bins spanning the
    range [-1,1].

    After quantization, a standard matrix-vector multiply is computed.
    However, if both `act_bit` and `weight_bit` are one, then the results
    are scaled to fit in the range [0,#inputs].

    Attributes
    ----------
    weight : Parameter
        Parameter object for weights for this block.
    bias : Parameter or None
        Parameter object for biases for this block or None if not using bias.

    Inputs:
        - **data**: tensor of shape (batch-size, in_units) if `flatten` is False,
          or else (batch-size, x1,...xn) which will be implicitly flattened to
          the latter by combining the x dimensions. dtype float32

          The number of input dimensions (after flattening) must be a multiple
          of the machine word size (e.g. 32 or 64).

    Outputs:
        - **out**: tensor with dtype float32 and shape (batch-size, units).
    """
    def __init__(self, units, in_units=0, flatten=True,
                 act_bit=1, prepend_act=True,
                 weight_bit=1, weight_initializer=None, weight_dtype=np.float32,
                 use_bias=False, bias_initializer='zeros',
                 wordsize = 64,
                 **kwargs):
        """Construct QDense block

        Parameters
        ----------
        units : int (positive)
            Specifies the number of outputs of this block.
        in_units : int (positive), optional
            Specifies the number of input units. If omitted will
            be inferred from data on first forward run. Must be
            a multiple of the machine word size (e.g. 32 or 64).
        flatten : bool, optional, default = True
            If true, the inputs will be implicitly flattened to
            two dimensions where first is the batch size.
        act_bit : int {1,2,4,8,16,32}, optional, default = 1
            The quantization bit width of the input activations.
        prepend_act : bool, optional, default = True
            When true, the block will implicitly prepend a
            `QActivation` block with specified `act_bit` and
            `backward_only` set to True.
        weight_bit : int {1,2,4,8,16,32}, optional, default = 1
            The quantization bit width of the weights.
        weight_initializer : str, optional
            Specifies the initializer to use for the weights.
            If omitted, it will default to the global optimizer.
        use_bias : bool, optional, default = False
            Specifies whether block should use a bias parameter.
            Bias is redundant if block is followed by a `BatchNorm`.
            Not supported when `act_bit` and `weight_bit` are one.
        bias_initializer : str, optional, default='zeros'
            Specifies the initializer to use for the biases, if any.
        weight_dtype : numpy.dtype or str, optional, default = np.float32
            Specifies representation of weights and biases. Currently only
            supports `float32`, and integral type matching `wordsize`
            (e.g. `int64` for 64 bit words). Integral types
            currently only supported when `act_bit` and `weight_bit` are one,
            in which case weights will be packed in the bits of the word.
            Backward mode (i.e. training) is only supported when using
            floating point weights.
        wordsize : int, optional, default=64
            Specifies the size in bits of the target machine word.
        **kwargs: see `mxnet.gluon.Block`
        """
        super(QDense, self).__init__(**kwargs)
        with self.name_scope():
            self._act_bit = _check_bit_arg('act_bit', act_bit)
            self._weight_bit = _check_bit_arg('weight_bit', weight_bit)
            self._prepend_act = prepend_act
            self._flatten = flatten
            self._units = units
            self._in_units = in_units

            if in_units % wordsize != 0:
                raise ValueError('in_units %d not a multiple of the wordsize %d' %
                                 (in_units, wordsize))

            strictly_binary = act_bit == 1 and weight_bit == 1

            weight_dtype = np.dtype(weight_dtype)

            if weight_dtype.type not in {np.float32, np.int32, np.int64}:
                raise ValueError('Unsupported weight_dtype %s' % weight_dtype.name)

            if weight_dtype.kind =='i':
                weights_per_item = weight_dtype.itemsize * 8
                if weights_per_item != wordsize:
                    raise ValueError('weight_dtype %s does not match wordsize %d' %
                                     (weight_dtype.name, wordsize))
                if not strictly_binary:
                    raise ValueError('weight_dtype %s only supported when act_bit and weight_bit '
                                     'are both one')
            else:
                weights_per_item = 1

            self.weight = self.params.get('weight', shape=(units, in_units / weights_per_item),
                                          init=weight_initializer,
                                          dtype=weight_dtype.type,
                                          allow_deferred_init=True)
            if use_bias:
                if strictly_binary:
                    raise NotImplementedError('bias not supported with binary weights and '
                                              'activations')

                self.bias = self.params.get('bias', shape=(units,),
                                            init=bias_initializer,
                                            allow_deferred_init=True)
            else:
                self.bias = None

    @property
    def units(self):
        """Number of output units from this block."""
        return self._units

    @property
    def in_units(self):
        """Number of input units to this block, if known.

        Will be zero if it is to be inferred from previous block.
        """
        return self._in_units

    @property
    def act_bit(self):
        """Bit width of quantization applied to activations"""
        return self._act_bit

    @property
    def weight_bit(self):
        """Bit width of quantization applied to weights"""
        return self._weight_bit

    @property
    def prepend_act(self):
        """If true, a `QActivation` with `backward_only=True` will be implicitly prepended."""
        return self._prepend_act

    @property
    def binarized_weights(self):
        """True if weights are stored in binary format.

        This is only supported when `act_bit` and `weight_bit` are both one.
        """
        return np.dtype(self.weight.dtype).kind == 'i'

    def hybrid_forward(self, F, x, weight, bias=None):
        if self._flatten:
            x = F.Flatten(x)
        if self._prepend_act:
            x = F.QActivation(x, act_bit=self._act_bit, backward_only=True)
        return F.QFullyConnected(x, weight, bias, no_bias=bias is None, num_hidden=self._units,
                                 act_bit=self._act_bit,
                                 weight_bit=self._weight_bit,
                                 binarized_weights_only=self.binarized_weights,
                                 name='fwd')

    def __repr__(self):
        s = '{name}({layout}, act_bit={act_bit}, weight_bit={weight_bit})'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        act_bit=self.act_bit,
                        weight_bit=self.weight_bit,
                        layout='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]))


class QConvBase(_Conv):
    r"""Base implementation for quantized convolutional blocks.

    This provides the shared parameters and implementation of the
    `QConv1D`, `QConv2D`, and `QConv3D` blocks.

    This block implements a quantized version of the standard Conv blocks.
    The quantization serves as the activation function for the layer.

    This will quantize the input activations using the act_bit parameter
    to determine the number of quantization bins. If act_bit is one,
    inputs are quantized to -1/+1, otherwise they are quantized to
    2^act_bit bins spanning the range [0,1].

    The weights are quantized to 2^weight_bit bins spanning the range [-1,1].

    After quantization, the standard convolution is computed.
    However, if both `act_bit` and `weight_bit` are one, then the results
    are scaled to fit in the range [0,#inputs].

    Attributes
    ----------
    weights : Parameter
        Parameter object for weights for this block.
    bias : Parameter or None
        Parameter object for biases for this block or None if not using bias.

    Inputs:
        - **data**: tensor of shape (batch-size, in_units) if `flatten` is False,
          or else (batch-size, x1,...xn) which will be implicitly flattened to
          the latter by combining the x dimensions. dtype float32

          The number of input dimensions (after flattening) must be a multiple
          of the machine word size (e.g. 32 or 64).

    Outputs:
        - **out**: tensor with dtype float32 and shape (batch-size, units).
    """

    def __init__(self, channels, kernel_size, strides, padding, dilation, groups, layout,
                 in_channels=0,
                 act_bit=1, prepend_act=True,
                 weight_bit=1, weight_initializer=None, weight_dtype=np.float32,
                 use_bias=False, bias_initializer='zeros',
                 scaling_factor=False, wordsize=64,
                 cudnn_tune=None, cudnn_off=False,
                 prefix=None, params=None):
        """Base constructor for QConv blocks

        Parameters
        ----------
        channels: int
            The number of output channels and number of convolutional filters.
        kernel_size: int or tuple/list of ints
            Dimensions of the convolution.
        strides: int or tuple/list of ints
            Specifies how much to move convolutional window across each dimension.
        padding: int or tuple/list of ints
            Specifies units of zero padding to add to both sides of each dimension.
        dilation: int or tuple/list of tins
            Specifies distance between adjacent pixels in convolution in each
            dimension.
        groups : int
            Controls the connections between inputs and outputs.
            At groups=1, all inputs are convolved to all outputs.
            At groups=2, the operation becomes equivalent to having two convolution
            layers side by side, each seeing half the input channels, and producing
            half the output channels, and both subsequently concatenated.
        layout: str
            Specifies ordering of dimensions for inputs and weights using 'D'
            for depth, 'H' for height', 'W' for width, 'C' for input channel,
            and 'N' for batch.
            Currently only supports NCW, NCHW and NCDHW layouts.
        in_channels: int, optional, default=0
            Number of input channels in the previous layer. If zero, it will
            be inferred from data when graph is first initialized. Must be
            a multiple of the wordsize
        act_bit : int {1,2,4,8,16,32}, optional, default = 1
            The quantization bit width of the input activations.
        prepend_act : bool, optional, default = True
            When true, the block will implicitly prepend a
            `QActivation` block with specified `act_bit` and
            `backward_only` set to True.
        weight_bit : int {1,2,4,8,16,32}, optional, default = 1
            The quantization bit width of the weights.
        weight_initializer : str, optional
            Specifies the initializer to use for the weights.
            If omitted, it will default to the global optimizer.
        use_bias : bool, optional, default = False
            Specifies whether block should use a bias parameter.
            Bias is redundant if block is followed by a `BatchNorm`.
            Not supported when `act_bit` and `weight_bit` are one.
        bias_initializer : str, optional, default='zeros'
            Specifies the initializer to use for the biases, if any.
        weight_dtype : numpy.dtype or str, optional, default = np.float32
            Specifies representation of weights and biases. Currently only
            supports `float32`, and integral type matching `wordsize`
            (e.g. `int64` for 64 bit words). Integral types
            currently only supported when `act_bit` and `weight_bit` are one,
            in which case weights will be packed in the bits of the word.
            Backward mode (i.e. training) is only supported when using
            floating point weights.
        wordsize : int, optional, default=64
            Specifies the size in bits of the target machine word.
        cudnn_tune: 'off', 'limited_workspace', or 'fastest'
            Whether to pick convolution algo by running performance test.
        cudnn_off: bool, optional, default=False
            Turn off cudnn for this layer.
         """
        super(QConvBase, self).__init__(channels, kernel_size, strides, padding, dilation, groups,
                                        layout,
                                        in_channels=in_channels,
                                        use_bias=use_bias,
                                        weight_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        op_name='QConvolution',
                                        prefix=prefix,
                                        params=params)

        strictly_binary = act_bit == 1 and weight_bit == 1

        weight_dtype = np.dtype(weight_dtype)

        if in_channels % wordsize != 0:
            raise ValueError('in_channels %d not a multiple of the wordsize %d' %
                             (in_channels, wordsize))

        if weight_dtype.type not in {np.float32, np.int32, np.int64}:
            raise ValueError('Unsupported weight_dtype %s' % weight_dtype.name)

        if weight_dtype.kind == 'i':
            weights_per_item = weight_dtype.itemsize * 8
            if weights_per_item != wordsize:
                raise ValueError('weight_dtype %s does not match wordsize %d' %
                                 (weight_dtype.name, wordsize))
            if not strictly_binary:
                raise ValueError('weight_dtype %s only supported when act_bit and weight_bit '
                                 'are both one')
        else:
            weights_per_item = 1

        self.weight.dtype = weight_dtype.type
        self.weight.shape = (channels, in_channels/weights_per_item,) + kernel_size

        if use_bias and strictly_binary:
            raise NotImplementedError('bias not supported with binary weights and '
                                      'activations')

        self._prepend_act = prepend_act
        self._kwargs.update(dict(act_bit=_check_bit_arg('act_bit', act_bit),
                                 weight_bit=_check_bit_arg('weight_bit', weight_bit),
                                 scaling_factor=scaling_factor,
                                 binarized_weights_only=weight_dtype.kind == 'i',
                                 cudnn_tune=cudnn_tune,
                                 cudnn_off=cudnn_off))

    @property
    def act_bit(self):
        """Bit width of quantization applied to activations"""
        return self._kwargs['act_bit']

    @property
    def weight_bit(self):
        """Bit width of quantization applied to weights"""
        return self._kwargs['weight_bit']

    @property
    def prepend_act(self):
        """If true, a `QActivation` with `backward_only=True` will be implicitly prepended."""
        return self._prepend_act

    @property
    def binarized_weights(self):
        """True if weights are stored in binary format.

        This is only supported when `act_bit` and `weight_bit` are both one.
        """
        return np.dtype(self.weight.dtype).kind == 'i'

    def _alias(self):
        return 'qconv'

    def __repr__(self):
        s = '{name}({mapping}, act_bit={act_bit}, weight_bit={weight_bit}, ' \
            'kernel={kernel}, stride={stride}'
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

    def hybrid_forward(self, F, x, weight, bias=None):
        if self._prepend_act:
            x = F.QActivation(x, act_bit=self.act_bit, backward_only=True)
        return super(QConvBase, self).hybrid_forward(F, x, weight, bias)


# TODO 1D convolution not yet supported
# class QConv1D(QConvBase):
#     """1-dimensional quantized convolution
#
#     See `QConvBase` for details.
#     """
#     def __init__(self, channels, kernel_size,
#                  strides=1, padding=0, dilation=1, groups=1, layout='NCW',
#                  in_channels=0,
#                  act_bit=1, prepend_act=True,
#                  weight_bit=1, weight_initializer=None, weight_dtype=np.float32,
#                  use_bias=False, bias_initializer='zeros',
#                  scaling_factor=False, wordsize=64,
#                  cudnn_tune=None, cudnn_off=False,
#                  prefix=None, params=None):
#         if isinstance(kernel_size, numeric_types):
#             kernel_size = (kernel_size,)
#         assert len(kernel_size) == 1, "kernel_size must be a number or a list of 1 ints"
#         super(QConv1D, self).__init__(channels, kernel_size, strides, padding, dilation, groups,
#                                       layout, in_channels,
#                                       act_bit, prepend_act,
#                                       weight_bit, weight_initializer, weight_dtype,
#                                       use_bias, bias_initializer,
#                                       scaling_factor, wordsize,
#                                       cudnn_tune, cudnn_off,
#                                       prefix, params)



class QConv2D(QConvBase):
    """2-dimensional quantized convolution

    See `QConvBase` for details.
    """
    def __init__(self, channels, kernel_size,
                 strides=(1,1), padding=(0,0), dilation=(1,1), groups=1, layout='NCHW',
                 in_channels=0,
                 act_bit=1, prepend_act=True,
                 weight_bit=1, weight_initializer=None, weight_dtype=np.float32,
                 use_bias=False, bias_initializer='zeros',
                 scaling_factor=False, wordsize=64,
                 cudnn_tune=None, cudnn_off=False,
                 prefix=None, params=None):
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        super(QConv2D, self).__init__(channels, kernel_size, strides, padding, dilation, groups,
                                      layout, in_channels,
                                      act_bit, prepend_act,
                                      weight_bit, weight_initializer, weight_dtype,
                                      use_bias,
                                      bias_initializer,
                                      scaling_factor, wordsize,
                                      cudnn_tune, cudnn_off,
                                      prefix, params)


# TODO 3D convolution not yet supported
# class QConv3D(QConvBase):
#     """3-dimensional quantized convolution
#
#     See `QConvBase` for details.
#     """
#     def __init__(self, channels, kernel_size,
#                  strides=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, layout='NCDHW',
#                  in_channels=0,
#                  act_bit=1, prepend_act=True,
#                  weight_bit=1, weight_initializer=None, weight_dtype=np.float32,
#                  use_bias=False, bias_initializer='zeros',
#                  scaling_factor=False, wordsize=64,
#                  cudnn_tune=None, cudnn_off=False,
#                  prefix=None, params=None):
#         if isinstance(kernel_size, numeric_types):
#             kernel_size = (kernel_size,)*3
#         assert len(kernel_size) == 3, "kernel_size must be a number or a list of 3 ints"
#         super(QConv3D, self).__init__(channels, kernel_size, strides, padding, dilation, groups,
#                                       layout, in_channels,
#                                       act_bit, prepend_act,
#                                       weight_bit, weight_initializer, weight_dtype,
#                                       use_bias, bias_initializer,
#                                       scaling_factor, wordsize,
#                                       cudnn_tune, cudnn_off,
#                                       prefix, params)




def _check_bit_arg(name, bit):
    """Verifies that act_bit/weight_bit keyword is a power of two and returns value"""
    if bit not in {1,2,4,8,16,32}:
        raise ValueError("Bad `%s` '%s' - not one of 1, 2, 4, 8, 16, or 32" % (name, bit))
    return bit