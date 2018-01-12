import argparse
import logging

import numpy as np

import mxnet as mx
from mxnet import autograd
from mxnet.callback import LogValidationMetricsCallback
from mxnet.gluon.data import ArrayDataset, DataLoader
from mxnet.gluon.data.vision import MNIST
from mxnet.gluon.loss import SoftmaxCELoss

from mxnet.gluon.nn import *
from mxnet.gluon.contrib.bnn import *
from mxnet.metric import Accuracy
from mxnet.model import BatchEndParam

BITW = 1
BITA = 1

logging.getLogger().setLevel(logging.INFO)

class LeNet(HybridBlock):
    def __init__(self, binary=True):
        super(LeNet, self).__init__(prefix='LeNet_')
        self.binary = binary

        with self.name_scope():
            self._layers = layers = HybridSequential(prefix='LeNet_')

            # first conv layer
            layers.add(
                Conv2D(channels=64, kernel_size=5, activation='tanh'),
                MaxPool2D(pool_size=2),
                BatchNorm())

            # second conv layer
            if binary:
                layers.add(
                    QActivation(act_bit=BITA, backward_only=True),
                    QConv2D(channels=64, kernel_size=5, act_bit=BITW, use_bias=False))
            else:
                layers.add(Conv2D(channels=64, kernel_size=5, use_bias=False))

            layers.add(
                BatchNorm(),
                MaxPool2D(pool_size=2))

            # first fully connected layer
            if binary:
                layers.add(
                    Flatten(),
                    QActivation(act_bit=BITA, backward_only=True),
                    QDense(1000, act_bit=BITW, use_bias=False))
            else:
                layers.add(Dense(1000, use_bias=False))

            layers.add(
                BatchNorm(),
                Activation('tanh'))

            # second fully connected layer
            layers.add(Dense(10))

    def hybrid_forward(self, F, data, labels):
        out = self._layers(data)
        return F.SoftmaxOutput(data=out, label=labels)


def mnist_transform(data,label):
    shape = data.shape
    batch_size = 1 if len(shape) == 3 else shape[0]
    return data.reshape((batch_size,1,28,28)).astype(np.float32) / 255, label.astype(np.float32)

def train_mnist(model, epochs=10, batch_size=200, file_prefix='mnist', debug=False):

    base_training_set = MNIST(train=True, transform=mnist_transform)
    training_set = ArrayDataset(*base_training_set[:50000])
    validation_set = ArrayDataset(*base_training_set[50000:])

    training_batches = DataLoader(training_set,
                                  batch_size=batch_size,
                                  shuffle=True)
    validation_batches = DataLoader(validation_set,
                                    batch_size=batch_size,
                                    shuffle=False)
    params = model.collect_params()
    params.initialize('Xavier')

    # export requires hyridized model with at least one forward run
    model.hybridize()
    model(training_set[:2][0], mx.ndarray.array(training_set[:2][1]))
    model.export(file_prefix)

    if debug:
        model.hybridize(False)

    trainer = mx.gluon.Trainer(params, 'adam', optimizer_params=dict(learning_rate=.01))

    metric = Accuracy(axis=0)
    speedometer = mx.callback.Speedometer(batch_size, 5)
    log_validation = LogValidationMetricsCallback()

    for epoch in range(epochs):
        metric.reset()

        nbatch = 0
        for images, labels in training_batches:
            with autograd.record():
                predictions = model(images, labels)

            predictions.backward()

            metric.update(labels, predictions)

            trainer.step(batch_size)

            speedometer(BatchEndParam(epoch=epoch, nbatch=nbatch, eval_metric=metric, locals={}))

            nbatch += 1

        metric.reset()
        for images, labels in validation_batches:
            predictions = model(images, labels)
            metric.update(labels, mx.ndarray.argmax(predictions, axis=1))

        log_validation(BatchEndParam(epoch=epoch, nbatch=0, eval_metric=metric,locals={}))

        params.save('%s-%04d.params' % (file_prefix, epoch+1))

def test_mnist(model, batch_size=200):
    test_set = ArrayDataset(*MNIST(train=False, transform=mnist_transform)[:])
    test_batches = DataLoader(test_set,
                              batch_size=batch_size,
                              shuffle=False)

    metric = Accuracy(axis=0)

    for images, labels in test_batches:
        predictions = model(images, labels)
        metric.update(labels, mx.ndarray.argmax(predictions, axis=1))

    print('Test %s=%s' % metric.get())


def main(args):
    ctx = mx.gpu(args.gpu_id) if args.gpu_id >= 0 else mx.cpu()

    with mx.Context(ctx):
        if args.predict:
            out = mx.sym.load('%s-symbol.json' % args.model)
            model = SymbolBlock(out, [mx.sym.var('data0'), mx.sym.var('data1')])
            model.collect_params().load('%s-%04d.params' % (args.model, args.epochs), mx.cpu())

            test_mnist(model, batch_size=args.batch_size)

        else:
            model = LeNet(binary=args.binary)

            train_mnist(model,
                        debug=args.debug,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        file_prefix=args.model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate generate annotations file from data source')

    parser.add_argument('--no-binary', dest='binary', action='store_false',
                        help='when training, do not used binarized model')
    parser.add_argument('--model', type=str, default='lenet-gluon',
                        help="name of model used for .json and .params files")
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1,
                        help='selected gpu device id (default is cpu)')
    parser.add_argument('--predict', dest='predict', action='store_true',default=False,
                        help='whether do the prediction, otherwise do the training')
    parser.add_argument('--epochs', dest='epochs', type=int, default=0,
                        help='# of epochs of training or epoch params to use for prediction')
    parser.add_argument('-B', '--batch-size', type=int, default=200, help='set the batch size')
    parser.add_argument('--debug', action='store_true', help='Debug mode - use imperative api')

    args = parser.parse_args()
    main(args)
