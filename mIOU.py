import math

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import Variable
from chainer import reporter
from chainer import cuda
import cupy
from chainercv import evaluations

class PixelwiseSigmoidClassifier(chainer.Chain):

    def __init__(self, predictor):
        super().__init__()
        with self.init_scope():
            # 学習対象のモデルをpredictorとして保持しておく
            self.predictor = predictor

    def __call__(self, x, t):
        # 学習対象のモデルでまず推論を行う
        y = self.predictor(x)

        # 5クラス分類の誤差を計算
        loss = F.softmax_cross_entropy(y, t)

        # 予測結果（0~1の連続値を持つグレースケール画像）を二値化し，
        # ChainerCVのeval_semantic_segmentation関数に正解ラベルと
        # 共に渡して各種スコアを計算
        #y = np.asarray(y > 0.5, dtype=np.int32)
        #y, t = y[:, 0, ...], t[:, 0, ...]
        #evals = evaluations.eval_semantic_segmentation(y, t)

        # 学習中のログに出力
        reporter.report({'loss': loss},
                         #'miou': evals['miou'],
                         #'pa': evals['pixel_accuracy']}, 
                         self)
        return loss