#!/usr/bin/env python

import chainer
import chainer.functions as F
import chainer.links as L

class UNet(chainer.Chain):
    def __init__(self, out_h, out_w, n_class=5):
        super(UNet, self).__init__()
        self.c0=L.Convolution2D(3, 32, 3, 1, 1)
        self.c1=L.Convolution2D(32, 64, 4, 2, 1)
        self.c2=L.Convolution2D(64, 64, 3, 1, 1)
        self.c3=L.Convolution2D(64, 128, 4, 2, 1)
        self.c4=L.Convolution2D(128, 128, 3, 1, 1)
        self.c5=L.Convolution2D(128, 256, 4, 2, 1)
        self.c6=L.Convolution2D(256, 256, 3, 1, 1)
        self.c7=L.Convolution2D(256, 512, 4, 2, 1)
        self.c8=L.Convolution2D(512, 512, 3, 1, 1)

        self.dc8=L.Deconvolution2D(1024, 512, 4, 2, 1)
        self.dc7=L.Convolution2D(512, 256, 3, 1, 1)
        self.dc6=L.Deconvolution2D(512, 256, 4, 2, 1)
        self.dc5=L.Convolution2D(256, 128, 3, 1, 1)
        self.dc4=L.Deconvolution2D(256, 128, 4, 2, 1)
        self.dc3=L.Convolution2D(128, 64, 3, 1, 1)
        self.dc2=L.Deconvolution2D(128, 64, 4, 2, 1)
        self.dc1=L.Convolution2D(64, 32, 3, 1, 1)
        self.dc0=L.Convolution2D(64, n_class, 3, 1, 1)

        self.bnc0=L.BatchNormalization(32)
        self.bnc1=L.BatchNormalization(64)
        self.bnc2=L.BatchNormalization(64)
        self.bnc3=L.BatchNormalization(128)
        self.bnc4=L.BatchNormalization(128)
        self.bnc5=L.BatchNormalization(256)
        self.bnc6=L.BatchNormalization(256)
        self.bnc7=L.BatchNormalization(512)
        self.bnc8=L.BatchNormalization(512)

        self.bnd8=L.BatchNormalization(512)
        self.bnd7=L.BatchNormalization(256)
        self.bnd6=L.BatchNormalization(256)
        self.bnd5=L.BatchNormalization(128)
        self.bnd4=L.BatchNormalization(128)
        self.bnd3=L.BatchNormalization(64)
        self.bnd2=L.BatchNormalization(64)
        self.bnd1=L.BatchNormalization(32)        
        self.n_class = n_class
        self.out_h = out_h
        self.out_w = out_w
        self.train = False

    def forward(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        e1 = F.relu(self.bnc1(self.c1(e0)))
        e2 = F.relu(self.bnc2(self.c2(e1)))
        del e1
        e3 = F.relu(self.bnc3(self.c3(e2)))
        e4 = F.relu(self.bnc4(self.c4(e3)))
        del e3
        e5 = F.relu(self.bnc5(self.c5(e4)))
        e6 = F.relu(self.bnc6(self.c6(e5)))
        del e5
        e7 = F.relu(self.bnc7(self.c7(e6)))
        e8 = F.relu(self.bnc8(self.c8(e7)))

        d8 = F.relu(self.bnd8(self.dc8(F.concat([e7, e8]))))
        del e7, e8
        d7 = F.relu(self.bnd7(self.dc7(d8)))
        del d8
        d6 = F.relu(self.bnd6(self.dc6(F.concat([e6, d7]))))
        del d7, e6
        d5 = F.relu(self.bnd5(self.dc5(d6)))
        del d6
        d4 = F.relu(self.bnd4(self.dc4(F.concat([e4, d5]))))
        del d5, e4
        d3 = F.relu(self.bnd3(self.dc3(d4)))
        del d4
        d2 = F.relu(self.bnd2(self.dc2(F.concat([e2, d3]))))
        del d3, e2
        d1 = F.relu(self.bnd1(self.dc1(d2)))
        del d2
        d0 = self.dc0(F.concat([e0, d1]))

        return d0