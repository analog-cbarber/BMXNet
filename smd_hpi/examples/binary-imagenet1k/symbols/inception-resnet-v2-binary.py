"""
Contains the definition of the Inception Resnet V2 architecture.		
As described in http://arxiv.org/abs/1602.07261.		
Inception-v4, Inception-ResNet and the Impact of Residual Connections		
on Learning		
Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi		
"""
import mxnet as mx

BITW = -1 # set in get_symbol
BITA = -1 # set in get_symbol

#======================== original conv block =========================#
def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type="relu", mirror_attr={}, with_act=True):
    conv = mx.symbol.Convolution(
        data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    if with_act:
        act = mx.symbol.Activation(
            data=bn, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return bn
#======================================================================#

#======================== binary conv block =========================#
def QConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type="relu", mirror_attr={}):
    bn = mx.symbol.BatchNorm(data=data, fix_gamma=False, eps=2e-5)
    qact = mx.sym.QActivation(data=bn, act_bit=BITA, backward_only=True)
    conv = mx.symbol.QConvolution(
        data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, act_bit=BITA, weight_bit=BITW, cudnn_off=False)
    bn2 = mx.symbol.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.9)
    lrelu = mx.symbol.LeakyReLU(data=bn2, act_type="leaky")
    return lrelu
#====================================================================#

def block35(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}):
    tower_conv = QConvFactory(net, 32, (1, 1))
    tower_conv1_0 = QConvFactory(net, 32, (1, 1))
    tower_conv1_1 = QConvFactory(tower_conv1_0, 32, (3, 3), pad=(1, 1))
    tower_conv2_0 = QConvFactory(net, 32, (1, 1))
    tower_conv2_1 = QConvFactory(tower_conv2_0, 48, (3, 3), pad=(1, 1))
    tower_conv2_2 = QConvFactory(tower_conv2_1, 64, (3, 3), pad=(1, 1))
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_1, tower_conv2_2])
    tower_out = QConvFactory(
        tower_mixed, input_num_channels, (1, 1))

    net = net + scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def block17(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}):
    tower_conv = QConvFactory(net, 192, (1, 1))
    tower_conv1_0 = QConvFactory(net, 129, (1, 1))
    tower_conv1_1 = QConvFactory(tower_conv1_0, 160, (1, 7), pad=(1, 2))
    tower_conv1_2 = QConvFactory(tower_conv1_1, 192, (7, 1), pad=(2, 1))
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
    tower_out = QConvFactory(
        tower_mixed, input_num_channels, (1, 1))
    net = net + scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def block8(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}):
    tower_conv = QConvFactory(net, 192, (1, 1))
    tower_conv1_0 = QConvFactory(net, 192, (1, 1))
    tower_conv1_1 = QConvFactory(tower_conv1_0, 224, (1, 3), pad=(0, 1))
    tower_conv1_2 = QConvFactory(tower_conv1_1, 256, (3, 1), pad=(1, 0))
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
    tower_out = QConvFactory(
        tower_mixed, input_num_channels, (1, 1))
    net = net + scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def repeat(inputs, repetitions, layer, *args, **kwargs):
    outputs = inputs
    for i in range(repetitions):
        outputs = layer(outputs, *args, **kwargs)
    return outputs


def get_symbol(num_classes=1000, bits_w=1, bits_a=1, **kwargs):
    global BITW, BITA
    BITW = bits_w
    BITA = bits_a

    data = mx.symbol.Variable(name='data')
    conv1a_3_3 = ConvFactory(data=data, num_filter=32,
                             kernel=(3, 3), stride=(2, 2))
    conv2a_3_3 = QConvFactory(conv1a_3_3, 32, (3, 3))
    conv2b_3_3 = QConvFactory(conv2a_3_3, 64, (3, 3), pad=(1, 1))
    maxpool3a_3_3 = mx.symbol.Pooling(
        data=conv2b_3_3, kernel=(3, 3), stride=(2, 2), pool_type='max')
    conv3b_1_1 = QConvFactory(maxpool3a_3_3, 80, (1, 1))
    conv4a_3_3 = QConvFactory(conv3b_1_1, 192, (3, 3))
    maxpool5a_3_3 = mx.symbol.Pooling(
        data=conv4a_3_3, kernel=(3, 3), stride=(2, 2), pool_type='max')

    tower_conv = QConvFactory(maxpool5a_3_3, 96, (1, 1))
    tower_conv1_0 = QConvFactory(maxpool5a_3_3, 48, (1, 1))
    tower_conv1_1 = QConvFactory(tower_conv1_0, 64, (5, 5), pad=(2, 2))

    tower_conv2_0 = QConvFactory(maxpool5a_3_3, 64, (1, 1))
    tower_conv2_1 = QConvFactory(tower_conv2_0, 96, (3, 3), pad=(1, 1))
    tower_conv2_2 = QConvFactory(tower_conv2_1, 96, (3, 3), pad=(1, 1))

    tower_pool3_0 = mx.symbol.Pooling(data=maxpool5a_3_3, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg')
    tower_conv3_1 = QConvFactory(tower_pool3_0, 64, (1, 1))
    tower_5b_out = mx.symbol.Concat(
        *[tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1])
    net = repeat(tower_5b_out, 10, block35, scale=0.17, input_num_channels=320)
    tower_conv = QConvFactory(net, 384, (3, 3), stride=(2, 2))
    tower_conv1_0 = QConvFactory(net, 256, (1, 1))
    tower_conv1_1 = QConvFactory(tower_conv1_0, 256, (3, 3), pad=(1, 1))
    tower_conv1_2 = QConvFactory(tower_conv1_1, 384, (3, 3), stride=(2, 2))
    net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, momentum=0.9)
    tower_pool = mx.symbol.Pooling(net, kernel=(
        3, 3), stride=(2, 2), pool_type='max')
    net = mx.symbol.Concat(*[tower_conv, tower_conv1_2, tower_pool])
    net = repeat(net, 20, block17, scale=0.1, input_num_channels=1088)
    tower_conv = QConvFactory(net, 256, (1, 1))
    tower_conv0_1 = QConvFactory(tower_conv, 384, (3, 3), stride=(2, 2))
    tower_conv1 = QConvFactory(net, 256, (1, 1))
    tower_conv1_1 = QConvFactory(tower_conv1, 288, (3, 3), stride=(2, 2))
    tower_conv2 = QConvFactory(net, 256, (1, 1))
    tower_conv2_1 = QConvFactory(tower_conv2, 288, (3, 3), pad=(1, 1))
    tower_conv2_2 = QConvFactory(tower_conv2_1, 320, (3, 3),  stride=(2, 2))
    net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, momentum=0.9)
    tower_pool = mx.symbol.Pooling(net, kernel=(
        3, 3), stride=(2, 2), pool_type='max')
    net = mx.symbol.Concat(
        *[tower_conv0_1, tower_conv1_1, tower_conv2_2, tower_pool])

    net = repeat(net, 9, block8, scale=0.2, input_num_channels=2080)
    net = block8(net, with_act=False, input_num_channels=2080)

    net = QConvFactory(net, 1536, (1, 1))
    net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, momentum=0.9)
    net = mx.symbol.Pooling(net, kernel=(
        1, 1), global_pool=True, stride=(2, 2), pool_type='avg')
    net = mx.symbol.Flatten(net)
    #net = mx.symbol.Dropout(data=net, p=0.2)
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    return softmax
