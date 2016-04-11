#!/usr/bin/env python
# encoding: utf-8
"""
COPYRIGHT

2016-02-11 - Created by Ehsan Hosseini-Asl

DCCA - MLP

"""

import argparse
import numpy as np
import os
import Queue
import threading
from PIL import Image
import pickle
import cPickle
import random
import sys
import time
import urllib
import theano
import theano.tensor as T
from theano.tensor import nnet
from theano.tensor.signal import downsample
import ipdb
from itertools import izip
import math
import scipy.io as sio

FLOAT_PRECISION = np.float32

#EHA: ADADELTA
def adadelta_updates(parameters, gradients, rho, eps):

    # create variables to store intermediate updates
    gradients_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=FLOAT_PRECISION),) for p in parameters ]
    deltas_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=FLOAT_PRECISION)) for p in parameters ]

    # calculates the new "average" delta for the next iteration
    gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in izip(gradients_sq,gradients) ]

    # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
    deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients) ]

    # calculates the new "average" deltas for the next step.
    deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in izip(deltas_sq,deltas) ]

    # Prepare it as a list f
    gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
    deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
    parameters_updates = [ (p,p - d) for p,d in izip(parameters,deltas) ]
    return gradient_sq_updates + deltas_sq_updates + parameters_updates


class ConvolutionLayer(object):
    ACT_TANH = 't'
    ACT_SIGMOID = 's'
    ACT_ReLu = 'r'
    ACT_SoftPlus = 'p'

    def __init__(self, rng, input, filter_shape, poolsize=(2,2), stride=None, if_pool=False, act=None, share_with=None,
                 tied=None, border_mode='valid'):
        self.input = input

        if share_with:
            self.W = share_with.W
            self.b = share_with.b

            self.W_delta = share_with.W_delta
            self.b_delta = share_with.b_delta

        elif tied:
            self.W = tied.W.dimshuffle(1,0,2,3)
            self.b = tied.b

            self.W_delta = tied.W_delta.dimshuffle(1,0,2,3)
            self.b_delta = tied.b_delta

        else:
            fan_in = np.prod(filter_shape[1:])
            poolsize_size = np.prod(poolsize) if poolsize else 1
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / poolsize_size)
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)

            self.W_delta = theano.shared(
                np.zeros(filter_shape, dtype=theano.config.floatX),
                borrow=True
            )

            self.b_delta = theano.shared(value=b_values, borrow=True)

        conv_out = nnet.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            border_mode=border_mode)

        #if poolsize:
        if if_pool:
            pooled_out = downsample.max_pool_2d(
                input=conv_out,
                ds=poolsize,
                st=stride,
                ignore_border=True)
            tmp = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            tmp = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        if act == ConvolutionLayer.ACT_TANH:
            self.output = T.tanh(tmp)
        elif act == ConvolutionLayer.ACT_SIGMOID:
            self.output = nnet.sigmoid(tmp)
        elif act == ConvolutionLayer.ACT_ReLu:
            self.output = tmp * (tmp>0)
        elif act == ConvolutionLayer.ACT_SoftPlus:
            self.output = T.log2(1+T.exp(tmp))
        else:
            self.output = tmp

        # store parameters of this layer
        self.params = [self.W, self.b]
        self.deltas = [self.W_delta, self.b_delta]

    def get_state(self):
        return self.W.get_value(), self.b.get_value()

    def set_state(self, state):
        self.W.set_value(state[0], borrow=True)
        self.b.set_value(state[1], borrow=True)

class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, share_with=None, activation=None):

        self.input = input

        if share_with:
            self.W = share_with.W
            self.b = share_with.b

            self.W_delta = share_with.W_delta
            self.b_delta = share_with.b_delta
        else:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == nnet.sigmoid:
                W_values *= 4

            self.W = theano.shared(value=W_values, name='W', borrow=True)

            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)

            self.W_delta = theano.shared(
                    np.zeros((n_in, n_out), dtype=theano.config.floatX),
                    borrow=True
                )

            self.b_delta = theano.shared(value=b_values, borrow=True)

        self.params = [self.W, self.b]

        self.deltas = [self.W_delta, self.b_delta]

        lin_output = T.dot(self.input, self.W) + self.b

        if activation == 'tanh':
            self.output = T.tanh(lin_output)
        elif activation == 'sigmoid':
            self.output = T.nnet.sigmoid(lin_output)
        elif activation == 'relu':
            self.output = 0.5 * (lin_output + abs(lin_output)) + 1e-9
        elif activation == 'cube_root':
            self.output = lin_output - ((1/3.)*self.output)**3
        else:
            self.output = lin_output

    def get_state(self):
        return self.W.get_value(), self.b.get_value()

    def set_state(self, state):
        self.W.set_value(state[0], borrow=True)
        self.b.set_value(state[1], borrow=True)

class softmaxLayer(object):
    def __init__(self, input, n_in, n_out):

        self.W = theano.shared(
            value=np.zeros(
                (n_in,n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.W_delta = theano.shared(
                np.zeros((n_in,n_out), dtype=theano.config.floatX),
                borrow=True
            )

        self.b_delta = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX),
            name='b',
            borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

        self.deltas = [self.W_delta, self.b_delta]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def get_state(self):
        return self.W.get_value(), self.b.get_value()

    def set_state(self, state):
        self.W.set_value(state[0], borrow=True)
        self.b.set_value(state[1], borrow=True)

class CCALayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=None):
        self.n_in = n_in
        self.n_out = n_out
        self.input = input
        self.activation = activation

        self.r1 = 0.001
        self.r2 = 0.001

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        if activation == 'tanh':
            self.output = T.tanh(lin_output)
        elif activation == 'sigmoid':
            self.output = nnet.sigmoid(lin_output)
        elif activation == 'relu':
            # self.output = T.maximum(lin_output, 0)
            self.output = 0.5 * (lin_output + abs(lin_output)) + 1e-9
        elif activation == 'cube_root':
            self.output = lin_output - ((1/3.)*self.output)**3
        else:
            self.output = lin_output

        self.params = [self.W]

    def correlation(self, H1, H2, m):
        H1bar = H1
        H2bar = H2
        SigmaHat12 = (1.0/(m-1))*T.dot(H1bar, H2bar.T)
        SigmaHat11 = (1.0/(m-1))*T.dot(H1bar, H1bar.T)
        SigmaHat11 = SigmaHat11 + self.r1*T.identity_like(SigmaHat11)
        SigmaHat22 = (1.0/(m-1))*T.dot(H2bar, H2bar.T)
        SigmaHat22 = SigmaHat22 + self.r2*T.identity_like(SigmaHat22)
        Tval = T.dot(SigmaHat11**(-0.5), T.dot(SigmaHat12, SigmaHat22**(-0.5)))
        corr = T.nlinalg.trace(T.dot(Tval.T, Tval))**(0.5)
        self.SigmaHat11 = SigmaHat11
        self.SigmaHat12 = SigmaHat12
        self.SigmaHat22 = SigmaHat22
        self.H1bar = H1bar
        self.H2bar = H2bar
        self.Tval = Tval
        return -1*corr

class DCCA(object):
    def __init__(self, h_size_net1=None, h_size_net2=None, act_func=None, learning_rate=0.0001):
        rng = np.random.RandomState(None)
        input1 = T.fmatrix('input1')
        input2 = T.fmatrix('input2')

        hidden1_net1 = HiddenLayer(rng,
                              input1,
                              h_size_net1[0],
                              h_size_net1[1],
                              activation=act_func)
        hidden2_net1 = HiddenLayer(rng,
                              hidden1_net1.output,
                              h_size_net1[1],
                              h_size_net1[2],
                              activation=act_func)
        CCA1 = CCALayer(rng,
                        input=hidden2_net1.output,
                        n_in=h_size_net1[2],
                        n_out=h_size_net1[3],
                        activation=act_func)

        self.net1_layers = [hidden1_net1, hidden2_net1, CCA1]
        self.net1_param = sum([layer.params for layer in self.net1_layers], [])

        # net#2
        hidden1_net2 = HiddenLayer(rng,
                              input2,
                              h_size_net2[0],
                              h_size_net2[1],
                              activation=act_func)
        hidden2_net2 = HiddenLayer(rng,
                              hidden1_net2.output,
                              h_size_net2[1],
                              h_size_net2[2],
                              activation=act_func)
        CCA2 = CCALayer(rng,
                        input=hidden2_net2.output,
                        n_in=h_size_net2[2],
                        n_out=h_size_net2[3],
                        activation=act_func)

        self.net2_layers = [hidden1_net2, hidden2_net2, CCA2]
        self.net2_param = sum([layer.params for layer in self.net2_layers], [])

        self.params = self.net1_param + self.net2_param

        # compute CCA correlation
        self.cost1 = CCA1.correlation(CCA1.output.T, CCA2.output.T, h_size_net1[-1])
        self.cost2 = CCA2.correlation(CCA1.output.T, CCA2.output.T, h_size_net2[-1])

        # compute gradient of cost to CCA1 layer
        U, V, D = theano.tensor.nlinalg.svd(CCA1.Tval)
        UVT = T.dot(U, V.T)
        Delta12 = T.dot(CCA1.SigmaHat11**(-0.5), T.dot(T.diag(UVT), CCA1.SigmaHat22**(-0.5)))
        UDUT = T.dot(U, T.dot(D, U.T))
        Delta11 = (-0.5) * T.dot(CCA1.SigmaHat11**(-0.5), T.dot(UDUT, CCA1.SigmaHat22**(-0.5)))
        grad_E_to_o = (1.0/8) * (2*T.dot(Delta11, CCA1.H1bar)+T.dot(Delta12, CCA1.H2bar))
        gparam_CCA1_W = T.dot(self.net1_layers[-2].output.T, (grad_E_to_o * (CCA1.output*(1-CCA1.output)).T).T)
        gparams1 = [T.grad(self.cost1, param) for param in self.net1_param[:-1]]
        gparams1.append(gparam_CCA1_W)

        # compute gradient of cost to CCA2 layer
        U, V, D = theano.tensor.nlinalg.svd(CCA2.Tval)
        UVT = T.dot(U, V.T)
        Delta12 = T.dot(CCA2.SigmaHat11**(-0.5), T.dot(T.diag(UVT), CCA2.SigmaHat22**(-0.5)))
        UDUT = T.dot(U, T.dot(D, U.T))
        Delta11 = (-0.5) * T.dot(CCA2.SigmaHat11**(-0.5), T.dot(UDUT, CCA2.SigmaHat22**(-0.5)))
        grad_E_to_o = (1.0/8) * (2*T.dot(Delta11, CCA2.H1bar)+ T.dot(Delta12, CCA2.H2bar))
        gparam_CCA2_W = T.dot(self.net2_layers[-2].output.T, (grad_E_to_o * (CCA2.output*(1-CCA2.output)).T).T)
        gparams2 = [T.grad(self.cost2, param) for param in self.net2_param[:-1]]
        gparams2.append(gparam_CCA2_W)

        updates1 = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.net1_param, gparams1)
            ]
        updates2 = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.net2_param, gparams2)
            ]

        self.grad_net = gparams1 + gparams2
        self.update_net = updates1 + updates2

        self.train = theano.function(
            inputs=[input1, input2],
            outputs=self.cost1,
            updates=self.update_net,
            name='train model'
        )

        self.forward = theano.function(
            inputs=[input1, input2],
            outputs=(CCA1.output, CCA2.output),
            name='forward'
        )

        self.test = theano.function(
            inputs=[input1, input2],
            outputs=(self.cost1, gparam_CCA1_W),
            name='test'
        )

    def save(self, filename):
        f = open(filename, 'w')
        for l in self.net1_layers:
            pickle.dump(l.get_state(), f, -1)
        for l in self.net2_layers:
            pickle.dump(l.get_state(), f, -1)
        f.close()

    def load(self, filename):
        f = open(filename, 'r')
        for l in self.net1_layers:
            l.set_state(pickle.load(f))
        for l in self.net2_layers:
            l.set_state(pickle.load(f))
        f.close()
        print 'model loaded from', filename

class MLP(object):
    def __init__(self, hidden_size=None, act_func=None):
        rng = np.random.RandomState(None)
        input = T.fmatrix('input')
        label = T.lvector('labels')

        hidden1 = HiddenLayer(rng,
                              input,
                              hidden_size[0],
                              hidden_size[1],
                              activation=act_func)

        hidden2 = HiddenLayer(rng,
                              hidden1.output,
                              hidden_size[1],
                              hidden_size[2],
                              activation=act_func)

        output_layer = softmaxLayer(input=hidden2.output,
                                    n_in=hidden_size[2],
                                    n_out=hidden_size[3])

        self.layers = [hidden1, hidden2, output_layer]
        self.params = sum([layer.params for layer in self.layers], [])

        self.cost = output_layer.negative_log_likelihood(label)
        self.grads = T.grad(self.cost, self.params)

        self.updates = adadelta_updates(parameters=self.params,
                                        gradients=self.grads,
                                        rho=0.95,
                                        eps=1e-6)

        self.error = output_layer.errors(label)
        self.y_pred = output_layer.y_pred
        self.prob = output_layer.p_y_given_x.max(axis=1)
        self.true_prob = output_layer.p_y_given_x[T.arange(label.shape[0]), label]
        self.p_y_given_x = output_layer.p_y_given_x
        self.train = theano.function(
            inputs=[input, label],
            outputs=(self.error, self.cost, self.y_pred, self.prob),
            updates=self.updates
        )

        self.forward = theano.function(
            inputs=[input, label],
            outputs=(self.error, self.y_pred, self.prob, self.true_prob, self.p_y_given_x)
        )

    def save(self, filename):
        f = open(filename, 'w')
        for l in self.layers:
            pickle.dump(l.get_state(), f, -1)
        f.close()

    def load(self, filename):
        f = open(filename)
        for l in self.layers:
            l.set_state(pickle.load(f))
        f.close()
        print 'model loaded from', filename

def do_train_dcca(model=None, input1=None, input2=None, dataset=None):
    loss = 0
    # loss_history = []
    epoch = 0
    progress_report = 10
    save_interval = 1800
    last_save = time.time()
    try:
        print 'training dcca'
        while True:
            model.forward(input1, input2)
            cost = model.train(input1, input2)
            loss +=cost
            epoch += 1
            if epoch % progress_report == 0:
                loss /= progress_report
                print '%d\t%g' % (epoch, loss)
                sys.stdout.flush()
                loss = 0
            if time.time() - last_save >= save_interval:
                filename = 'dcca_model%d.pkl' % dataset
                model.save(filename)
                print 'model saved to', filename
                break
    except KeyboardInterrupt:
        filename = 'dcca_model%d.pkl' % dataset
        model.save(filename)
        print 'model saved to', filename

def do_train_classifier(dcca_model=None, classifier_model=None, input1= None, input2=None, target=None, dataset=None):
    loss = 0
    epoch = 0
    progress_report = 10
    save_interval = 1800
    last_save = time.time()
    try:
        print 'training classifier'
        print 'epoch\tloss\terror'
        while True:
            dcca_out1, dcca_out2 = dcca_model.forward(input1, input2)
            error, cost, pred, prob = classifier_model.train(dcca_out2, target)
            loss +=cost
            epoch += 1
            if epoch % progress_report == 0:
                loss /= progress_report
                print '%d\t%.6f\t%.2f' % (epoch, loss, error)
                sys.stdout.flush()
                loss = 0
            if time.time() - last_save >= save_interval:
                filename = 'mlp_model%d.pkl' % dataset
                classifier_model.save(filename)
                print 'model saved to', filename
                break
    except KeyboardInterrupt:
        filename = 'mlp_model%d.pkl' % dataset
        classifier_model.save(filename)
        print 'model saved to', filename

def do_test_dcca_mlp(dcca_model=None, classifier_model=None, input1= None, input2=None, target=None):
    print 'testing dcca_mlp classifier'
    dcca_out1, dcca_out2 = dcca_model.forward(input1, input2)
    error, y_pred, prob, true_prob, p_y_given_x = classifier_model.forward(dcca_out2, target)
    print '\nerror', error
    print 'label\tpred\tprob'
    for label, pred, pb in zip(target, y_pred, prob):
        print '%d\t%d\t%.2f' % (label, pred, pb)

def ProcessCommandLine():
    parser = argparse.ArgumentParser(description='train DCCA for MLP')
    parser.add_argument('-m', '--model',
                        help='start with this model')
    parser.add_argument('-set', '--dataset', type=int, default=None,
                        help='training on dataset')
    parser.add_argument('-tr', '--do_train', action='store_true',
                         help='do training')
    parser.add_argument('-ts', '--do_test', action='store_true',
                        help='do testing')
    parser.add_argument('-rd', '--do_reduce', action='store_true',
                        help='reduce features')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-9,
                        help='learning rate step')
    parser.add_argument('-act', '--activation', type=str, default='sigmoid',
                        help='activation function')
    parser.add_argument('-batch', type=int, default=1,
                        help='batch size')
    args = parser.parse_args()
    return args.model, args.dataset, args.do_train, args.do_test, args.do_reduce, args.learning_rate, args.activation, args.batch

def main():
    model, dataset, do_train, do_test, do_reduce, learning_rate, activation, batchsize = ProcessCommandLine()
    print 'learning rate:', learning_rate
    print 'activation:', activation
    print 'training on dataset#', dataset
    if do_reduce:
        print 'reduce features'

    sys_dir = os.getcwd()
    dir = sys_dir+'/data/feature1' if dataset==1 else sys_dir+'/data/feature2'
    data1_x = sio.loadmat(dir+'-1/traindata_brain_Euc%sL_bins_Norm.mat' % ('O' if dataset==1 else 'F'))
    data1_y = sio.loadmat(dir+'-1/trainLables_brain_Euc%sL_bins_Norm.mat' % ('O' if dataset==1 else 'F'))
    data1_x = np.asanyarray(data1_x['trainData'], dtype=np.float32)
    data1_y = data1_y['trainLabels'][0]
    data1_y[data1_y==1] = 0
    data1_y[data1_y==2] = 1
    data2_x = sio.loadmat(dir+'-2/traindata_brain_KN%sL_bins_Norm.mat' % ('O' if dataset==1 else 'F'))
    data2_y = sio.loadmat(dir+'-2/trainLables_brain_KN%sL_bins_Norm.mat' % ('O' if dataset==1 else 'F'))
    data2_x = np.asanyarray(data2_x['trainData'], dtype=np.float32)
    data2_y = data2_y['trainLabels'][0]
    data2_y[data2_y==1] = 0
    data2_y[data2_y==2] = 1

    if do_reduce:
        #skipping each 100 feature
        tmp1_x = np.empty((data1_x.shape[0]/100+1, data1_x.shape[1]), dtype=FLOAT_PRECISION)
        for idx, i in enumerate(xrange(0, data1_x.shape[0], 100)):
            tmp1_x[idx, :] = data1_x[i, :]
        data1_x = tmp1_x
        #skipping each 100 feature
        tmp2_x = np.empty((data2_x.shape[0]/100+1, data2_x.shape[1]), dtype=FLOAT_PRECISION)
        # ipdb.set_trace()
        for idx, i in enumerate(xrange(0, data2_x.shape[0], 100)):
            tmp2_x[idx, :] = data2_x[i, :]
        data2_x = tmp2_x

    input1_size, n_sample = data1_x.shape
    input2_size, n_sample = data2_x.shape
    n_output = np.unique(data1_y).shape[0]

    hidden_size_1 = [input1_size, 200, 100, 10]
    hidden_size_2 = [input2_size, 200, 100, 10]
    dcca = DCCA(h_size_net1=hidden_size_1,
                h_size_net2=hidden_size_2,
                learning_rate=learning_rate,
                act_func=activation)
    # ipdb.set_trace()
    classifier = MLP(hidden_size=[hidden_size_1[-1], 10, 5, n_output])

    # ipdb.set_trace()
    if do_train:
        do_train_dcca(model=dcca,
                      input1=data1_x.T,
                      input2=data2_x.T,
                      dataset=dataset)
        # ipdb.set_trace()
        do_train_classifier(dcca_model=dcca,
                            classifier_model=classifier,
                            input1=data1_x.T,
                            input2=data2_x.T,
                            target=data1_y,
                            dataset=dataset)
    elif do_test:
        dcca.load('dcca_model%d.pkl'%dataset)
        classifier.load('mlp_model%d.pkl'%dataset)
        do_test_dcca_mlp(dcca_model=dcca,classifier_model=classifier,input1=data1_x.T, input2=data2_x.T, target=data1_y)

if __name__ == '__main__':
    sys.exit(main())
