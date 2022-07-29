import imageio
from glob import glob
import os
import shutil
import numpy as np
from sklearn.utils import shuffle
import time
import speechpy

import kaldiio

from constants import *
import common


def has_uids(uids):
    for class in CLASSES:
        for gender in GENDERS:
            if len(uids[class][gender]) == 0:
                return False

    return True

def normalize_fb(spectrogram, output_shape=None):

    # Mean and Variance Normalization
    spectrogram = speechpy.processing.cmvn(
        spectrogram,
        variance_normalization=True)

    # MinMax Scaler, scale values between (0,1)
    normalized = (
        (spectrogram - np.min(spectrogram)) /
        (np.max(spectrogram) - np.min(spectrogram))
    )

    # Rotate 90deg
    normalized = np.swapaxes(normalized, 0, 1)

    # Reshape, tensor 3d
    if(output_shape is None):
        (height, width) = normalized.shape
        output_shape = (height, width, 1)

    normalized = normalized.reshape(output_shape)

    assert normalized.dtype == 'float32'
    assert np.max(normalized) == 1.0
    assert np.min(normalized) == 0.0

    return normalized


def write_fold(
        fold_uids,
        input_dir,
        input_ext,
        output_dir,
        group,
        fold_index,
        input_shape,
        output_shape):

    # find files for given uids
    fold_files = []
    for fold_uid in fold_uids:
        filename = '*{uid}*{extension}'.format(
            uid=fold_uid,
            extension=input_ext)
        fold_files.extend(glob(os.path.join(input_dir, filename)))

    #fold_files = sorted(fold_files)
    fold_files = shuffle(fold_files, random_state=SEED)
    #print(fold_files)
    metadata = []

    # create a file array
    filename = "{group}_data.fold{index}.npy".format(
        group=group, index=fold_index)
    features = np.memmap(
        os.path.join(output_dir, filename),
        dtype=DATA_TYPE,
        mode='w+',
        shape=(len(fold_files),) + output_shape)

    # append data to a file array
    # append metadata to an array
    for index, fold_file in enumerate(fold_files):
        #print(fold_file)
        filename = common.get_filename(fold_file)
        class = filename.split('_')[0]
        gender = filename.split('_')[1]
        key = (filename.split('_')[2]).split('.')[0]

        odata = np.load(fold_file)['data']
        #key, odata = next(kaldiio.load_ark(fold_file))
        #data = odata[...,3:43]

        data = None
        if (odata.shape != input_shape):
            deltax = input_shape[0] - odata.shape[0]
            deltay = input_shape[1] - odata.shape[1]
            assert deltax >= 0
            assert deltay >= 0
            data = np.pad(odata, ((0, deltax), (0, deltay)), 'constant', constant_values=0)
            #print("\t(ORIG: {}: {} {} ({}))".format(fold_file, odata.ndim, odata.shape, key));
        else:
            data = odata
        print("{}: {} {} {} {} {} ({})".format(fold_file, data.ndim, data.shape, data.size, np.min(data), np.max(data), key));
        #print("{}: {} {} ({})".format(fold_file, data.ndim, data.shape, key));

        assert data.shape == input_shape
        assert data.dtype == DATA_TYPE
        #print(data[0,])

        features[index] = normalize_fb(data, output_shape)
        #print(features[index][:,0])
        metadata.append((class, gender, filename))

    assert len(metadata) == len(fold_files)

    # store metadata in a file
    filename = "{group}_metadata.fold{index}.npy".format(
        group=group,
        index=fold_index)
    np.save(
        os.path.join(output_dir, filename),
        metadata)

    # flush changes to a disk
    features.flush()
    del features

    return len(fold_files)

def generate_fold(
        uids,
        input_dir,
        input_ext,
        output_dir,
        group,
        fold_index,
        input_shape,
        output_shape):

    #print('UIDS: %s\n' % uids)
    # pull uid for each a class, gender pair
    fold_uids = []
    for class in CLASSES:
        for gender in GENDERS:
            fold_uids.append(uids[class][gender].pop())

    #print(fold_uids)

    return write_fold(
                fold_uids,
                input_dir,
                input_ext,
                output_dir,
                group,
                fold_index,
                input_shape,
                output_shape)

def generate_folds(
        input_dir,
        input_ext,
        output_dir,
        group,
        input_shape,
        output_shape,
        num_folds=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = glob(os.path.join(input_dir, '*' + input_ext))

    uids = common.group_uids(files)
    #print(uids)
    #print("\n")
    num_files_in_folds = 0
    fold_index = 0

    if(num_folds < 2):
        while has_uids(uids):
            fold_index += 1
            print("[{group}] Fold {index}".format(group=group, index=fold_index))
            num_files_in_folds += generate_fold(
                                        uids,
                                        input_dir,
                                        input_ext,
                                        output_dir,
                                        group,
                                        fold_index,
                                        input_shape,
                                        output_shape)
    else:
        batchList = common.divide_uids(uids, num_folds)

        for i, batch in enumerate(batchList):
                fold_index = i+1
                print("[{group}] Fold {index}".format(group=group, index=fold_index))

                num_files_in_folds += write_fold(
                                            batch,
                                            input_dir,
                                            input_ext,
                                            output_dir,
                                            group,
                                            fold_index,
                                            input_shape,
                                            output_shape)

    print('Total %d %s files of which %d written to %d folds' %
    (len(files), group, num_files_in_folds, fold_index))



if __name__ == "__main__":
    start = time.time()

    out_dir = os.path.join(common.EXPTS_INT, 'folds')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)

    # fb
    #print('NUM_FOLDS=%d' % NUM_FOLDS)
    generate_folds(
        os.path.join(common.FEATS_DIST, 'test'),
        '.fb.npz',
        output_dir=os.path.join(common.EXPTS_INT, 'folds'),
        group='test',
        input_shape=(WIDTH, FB_HEIGHT),
        output_shape=(FB_HEIGHT, WIDTH, COLOR_DEPTH),
        num_folds=NUM_FOLDS
    )
    generate_folds(
        os.path.join(common.FEATS_DIST, 'train'),
        '.fb.npz',
        output_dir=os.path.join(common.EXPTS_INT, 'folds'),
        group='train',
        input_shape=(WIDTH, FB_HEIGHT),
        output_shape=(FB_HEIGHT, WIDTH, COLOR_DEPTH),
        num_folds=NUM_FOLDS
    )

    end = time.time()
    print("It took [s]: ", end - start)
