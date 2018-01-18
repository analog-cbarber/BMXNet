"""
Reference:

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.
"""
import mxnet as mx

eps = 2e-5
bn_mom = 0.9
fix_gamma = False

BITW = -1 # set in get_symbol
BITA = -1 # set in get_symbol

def get_symbol(num_classes, bits_w=1, bits_a=1, **kwargs):
    global BITW, BITA
    BITW = bits_w
    BITA = bits_a

    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution(
        data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96, name="convolution0")
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu1 = mx.symbol.Activation(data=bn1, act_type="relu")
    pool1 = mx.symbol.Pooling(
        data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))

    # stage 2
    act_q2 = mx.sym.QActivation(data=pool1,  act_bit=BITA, backward_only=True)
    conv2 = mx.symbol.QConvolution(
        data=act_q2, kernel=(5, 5), pad=(2, 2), num_filter=256, act_bit=BITA, weight_bit=BITW, name="convolution1")
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    pool2 = mx.symbol.Pooling(data=bn2, kernel=(3, 3), stride=(2, 2), pool_type="max")

    # stage 3    
    act_q3 = mx.sym.QActivation(data=pool2,  act_bit=BITA, backward_only=True)
    conv3 = mx.symbol.QConvolution(
        data=act_q3, kernel=(3, 3), pad=(1, 1), num_filter=384, act_bit=BITA, weight_bit=BITW, name="convolution2")
    bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)

    act_q4 = mx.sym.QActivation(data=bn3,  act_bit=BITA, backward_only=True)
    conv4 = mx.symbol.QConvolution(
        data=act_q4, kernel=(3, 3), pad=(1, 1), num_filter=384, act_bit=BITA, weight_bit=BITW, name="convolution3")
    bn4 = mx.sym.BatchNorm(data=conv4, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    
    act_q5 = mx.sym.QActivation(data=bn4,  act_bit=BITA, backward_only=True)
    conv5 = mx.symbol.QConvolution(
        data=act_q5, kernel=(3, 3), pad=(1, 1), num_filter=256, act_bit=BITA, weight_bit=BITW, name="convolution4")    
    bn5 = mx.sym.BatchNorm(data=conv5, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    pool3 = mx.symbol.Pooling(data=bn5, kernel=(3, 3), stride=(2, 2), pool_type="max")
   
    # stage 4
    flatten = mx.symbol.Flatten(data=pool3)
    act_fc1 = mx.sym.QActivation(data=flatten,  act_bit=BITA, backward_only=True)
    fc1 = mx.symbol.QFullyConnected(data=act_fc1, num_hidden=4096, act_bit=BITA, weight_bit=BITW)
    bn6 = mx.sym.BatchNorm(data=fc1, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu6 = mx.symbol.Activation(data=bn6, act_type="relu")

    # stage 5
    act_fc2 = mx.sym.QActivation(data=relu6,  act_bit=BITA, backward_only=True)
    fc2 = mx.symbol.QFullyConnected(data=act_fc2, num_hidden=4096, act_bit=BITA, weight_bit=BITW)
    bn7 = mx.sym.BatchNorm(data=fc2, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom)
    relu7 = mx.symbol.Activation(data=bn7, act_type="relu")
    
    # stage 6
    fc3 = mx.symbol.FullyConnected(data=relu7, num_hidden=num_classes, name="fullyconnected2")
    softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
    return softmax
