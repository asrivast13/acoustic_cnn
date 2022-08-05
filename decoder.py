import os
import sys
import features
import folds
import cnnbc
from sklearn import preprocessing
from sklearn.metrics import classification_report
from keras.models import Model, load_model, Sequential
import common
import numpy as np

class CNNBC_Decoder:
    def __init__(self, modelFileName):
        assert os.path.exists(modelFileName)

        self.model = load_model(modelFileName)
        #label_binarizer_, clazzes_ = common.build_label_binarizer()
        #self.label_binarizer = label_binarizer_
        #self.clazzes = clazzes_

        model_input_shape = self.model.layers[0].input_shape
        assert model_input_shape[-1] == 1

        self.NumFeats  = model_input_shape[1]
        self.SeqLength = model_input_shape[2]
        self.output_shape = model_input_shape[1:]
        #print(self.output_shape)

    def process(self, audioFile, threshold=0.5):

        signal, sample_rate = features.read_audio(audioFile)
        assert len(signal) > 0
        assert sample_rate >= 8000

        fb = features.generate_fb_and_mfcc(signal, sample_rate)
        fb = fb.astype('float32', copy=False)
        #print(fb.shape)
        assert self.NumFeats >= fb.shape[1], ("Feature dimension %d does not match model dimensionality %d" % (fb.shape[1], self.NumFeats))

        feats = fb
        if fb.shape != (self.SeqLength, self.NumFeats):
            if fb.shape[0] > self.SeqLength:
                feats = fb[0:self.SeqLength,...]
            else:
                deltax = self.SeqLength - fb.shape[0]
                deltay = self.NumFeats  - fb.shape[1]
                assert deltax >= 0
                assert deltay >= 0
                feats = np.pad(fb, ((0, deltax), (0, deltay)), 'constant', constant_values=0)

        assert feats.shape == (self.SeqLength, self.NumFeats)
        out_shape = (1, ) + self.output_shape
        input = folds.normalize_fb(feats, out_shape)
        results = self.model.predict(input, verbose=False)
        return results[0][0]

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='test model on audio file')
    parser.add_argument(
        '--audio',
        '-a',
        dest='audio',
        type=str,
        help='run model on single audio file')

    parser.add_argument(
        '--threshold',
        '-t',
        dest='threshold',
        type=float, default=common.THRESHOLD,
        help='score threshold for performance evaluation')

    parser.add_argument(
        '--model',
        '-m',
        dest='model',
        type=str,
        required=True,
        help='model file name to use to load state')

    args = parser.parse_args()

    decoder = CNNBC_Decoder(args.model)

    if args.audio:
        print(decoder.process(args.audio))
