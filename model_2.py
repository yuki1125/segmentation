import math

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import Variable

class FCNN(chainer.Chain):

    def __init__(self, out_h, out_w, n_class=5):
        super().__init__()
        with self.init_scope():
            self.conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1)
            self.conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1)

            self.conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1)
            self.conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1)

            self.conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1)
            self.conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1)

            self.conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1)
            self.conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1)

            self.conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1)

            self.pool3=L.Convolution2D(256, n_class, 1, stride=1, pad=0)
            self.pool4=L.Convolution2D(512, n_class, 1, stride=1, pad=0)
            self.pool5=L.Convolution2D(512, n_class, 1, stride=1, pad=0)

            self.upsample4=L.Deconvolution2D(n_class, n_class, ksize= 4, stride=2, pad=1)
            self.upsample5=L.Deconvolution2D(n_class, n_class, ksize= 8, stride=4, pad=2)
            self.upsample6=L.Deconvolution2D(n_class, n_class, ksize=16, stride=8, pad=4)
        self.n_class = n_class
        self.out_h = out_h
        self.out_w = out_w
        self.train = False

    def forward(self, x):
        h = F.relu(self.conv1_2(F.relu(self.conv1_1(x))))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv2_2(F.relu(self.conv2_1(h))))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(h))))))
        p3 = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(p3))))))
        p4 = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv5_3(F.relu(self.conv5_2(F.relu(self.conv5_1(p4))))))
        p5 = F.max_pooling_2d(h, 2, stride=2)

        p3 = self.pool3(p3)
        p4 = self.upsample4(self.pool4(p4))
        p5 = self.upsample5(self.pool5(p5))

        h = p3 + p4 + p5
        output = self.upsample6(h)
        return output