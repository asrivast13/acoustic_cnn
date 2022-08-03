from constants import *

from glob import glob
import time
import re

import statistics
# supress all warnings (especially matplotlib warnings)
import warnings
warnings.filterwarnings("ignore")

# RANDOMNESS
# https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

import os
os.environ['PYTHONHASHSEED'] = '0'

import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

# supress tensorflow debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# disable auto tune
# https://github.com/tensorflow/tensorflow/issues/5048
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

import tensorflow as tf
#import tensorflow.compat.v1 as tf
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)
from keras import backend as K
tf.compat.v1.set_random_seed(SEED)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

from sklearn import preprocessing
from sklearn.metrics import classification_report

from keras.models import Model, load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten
from keras.layers import Dropout, Input, Activation
#from keras.optimizers import Nadam, SGD
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from keras import regularizers

import common
tf.compat.v1.disable_eager_execution()


def build_model(input_shape):
    model = Sequential()

    # 40x1000

    model.add(Conv2D(
        16,
        (3, 3),
        strides=(1, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(0.001),
        input_shape=input_shape))
    model.add(Activation('elu'))
    #model.add(BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # 20x500

    model.add(Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('elu'))
    #model.add(BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # 10x250

    model.add(Conv2D(
        64,
        (3, 3),
        strides=(1, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('elu'))
    #model.add(BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # 5x125

    #model.add(Conv2D(
    #    128,
    #    (3, 5),
    #    strides=(1, 1),
    #    padding='same',
    #    kernel_regularizer=regularizers.l2(0.001)))
    #model.add(Activation('elu'))
    #model.add(MaxPooling2D(pool_size=(3, 5), strides=(1, 5), padding='same'))

    # 5x25

    model.add(Conv2D(
        256,
        (3, 5),
        strides=(1, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('elu'))
    #model.add(BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None))
    model.add(MaxPooling2D(pool_size=(3, 5), strides=(1, 5), padding='same'))
    model.add(AveragePooling2D(
        pool_size=(5, 5),
        strides=(5, 5),
        padding='same'))

    # 1x1
    model.add(BatchNormalization(epsilon=1e-06, momentum=0.0, weights=None))

    model.add(Flatten())

    model.add(Dense(
        32,
        activation='elu',
        kernel_regularizer=regularizers.l2(0.001)))

    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.0, nesterov=False)

    model.compile(
        loss='binary_crossentropy',
        optimizer=sgd,
        metrics=['accuracy'])

    return model

def decode_and_evaluate(modelFileName,
                        foldsFolder,
                        group,
                        input_shape,
                        threshold=0.5):
    start = time.time()
    model = load_model(modelFileName)
    print("Model loading took a total of: %.2fsecs\n" % (time.time()-start))

    start = time.time()
    label_binarizer, clazzes = common.build_label_binarizer()

    numFolds = len(glob(os.path.join(foldsFolder, ("%s_metadata.fold*.npy" % group))))
    assert numFolds > 0
    fold_indexes = list(range(1, numFolds + 1))

    test_labels, test_features, test_metadata = common.load_data(
        label_binarizer, foldsFolder, group, fold_indexes, input_shape)

    print("Data loading took a total of: %.2fsecs\n" % (time.time()-start))

    start = time.time()
    scores = model.evaluate(test_features, test_labels, verbose=0)
    delta = (time.time()-start)
    print("Model evaluation took a total of: %.2fsecs; Avg: %.6fsecs/file, Accuracy: %.4f\n" %  (delta, (delta/len(test_labels)), scores[1]))

    start = time.time()
    common.test(test_labels, test_features, test_metadata, model, clazzes, 'eval', threshold)
    print("Model testing and scoring took a total of: %.2fsecs\n" % (time.time()-start))

def train_and_validate( foldsFolder,
                        input_shape,
                        outModelFileName,
                        numEpochs=20,
                        batchSize=8,
                        doKfoldXValidation=False,
                        createModelRef=None,
                        threshold=0.0):

    accuracies = []
    numFolds = len(glob(os.path.join(foldsFolder, "train_metadata.fold*.npy")))
    numKfoldIterations = 1
    if doKfoldXValidation:
        numKfoldIterations = numFolds
        print("Starting %d fold cross-validation ..." % numKfoldIterations)

    if createModelRef is None:
	createModelRef = build_model

    modelFileName = outModelFileName

    generator = common.train_generator(
        numFolds, foldsFolder, input_shape, max_iterations=numKfoldIterations)

    first = True
    for (train_labels,
         train_features,
         test_labels,
         test_features,
         test_metadata,
         clazzes) in generator:

        model = createModelRef(input_shape)
        if first:
            model.summary()
            first = False

        checkpoint = ModelCheckpoint(
            modelFileName,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            mode='min')

        checkpoint2 = ModelCheckpoint(
            modelFileName + ".mva.h5",
            monitor='val_accuracy',
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode='max')

        earlystop = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=3,
            verbose=1,
            mode='auto')

        model.fit(
            train_features,
            train_labels,
            epochs=numEpochs,
            callbacks=[checkpoint, checkpoint2, earlystop],
            verbose=1,
            validation_data=(test_features, test_labels),
            batch_size=batchSize)

        model = load_model(modelFileName+".mva.h5")

        scores = model.evaluate(test_features, test_labels, verbose=0)
        accuracy = scores[1]

        print('Accuracy:', accuracy)
        accuracies.append(accuracy)

        common.test(
            test_labels,
            test_features,
            test_metadata,
            model,
            clazzes,
            'Dev',
            threshold)

    accuracies = np.array(accuracies)

    print('\n## Summary\n')
    print("Mean: {mean}, Std {std}".format(
        mean=accuracies.mean(),
        std=accuracies.std()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument(
        '--test',
        '--evaluate',
        '-e',
        dest='test',
        action='store_true',
        help='test the previously trained model against the test set')

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
        help='model file name to use to save/load state')

    parser.set_defaults(test=False, threshold=common.THRESHOLD)

    args = parser.parse_args()

    if args.model:
        modelFileName = args.model

    modelFileName = os.path.join(common.MODEL_DIST, 'model.h5')
    foldsFolder   = os.path.join(common.EXPTS_INT, 'folds')
    input_shape = (FB_HEIGHT, WIDTH, COLOR_DEPTH)

    if args.test:
        decode_and_evaluate(modelFileName,
                            foldsFolder,
                            'test',
                            input_shape,
                            args.threshold)
    else:
        train_and_validate( foldsFolder,
                            input_shape,
                            outModelFileName=modelFileName,
                            numEpochs=common.NUM_EPOCHS,
                            batchSize=common.BATCH_SIZE,
                            doKfoldXValidation=common.DO_K_FOLD_X_VALIDATION,
                            createModelRef=build_model,
                            threshold=args.threshold)
