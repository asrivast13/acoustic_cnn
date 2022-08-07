# Databricks notebook source

# MAGIC %md # Initialization

# COMMAND ----------

import sys
import os
import shutil
import glob
sys.path.append("/tmp/")
sys.path.append("/tmp/acoustic_cnn")

ORIG_AUDIO_DIR = '/dbfs/FileStore/AMD/audio'
DATASET_DIST   = '/dbfs/FileStore/AMD/input'
#DATASET_DIST   = '/tmp/AMD/input'
FEATS_DIST     = '/tmp/AMD/feats'
EXPTS_INT      = '/tmp/AMD'
MODEL_DIST     = '/tmp/AMD'
OUT_ARTIFACTS  = '/dbfs/FileStore/AMD/expts2/'

SEED = 42

SEG_DURATION = 2 # secs
FB_HEIGHT = 40  # filter banks
WIDTH = 198
COLOR_DEPTH = 1
INPUT_SHAPE = (FB_HEIGHT, WIDTH, COLOR_DEPTH)
NUM_FOLDS=10

DATA_TYPE = 'float32'
DATA_KEY = 'data'
CLASSES = ['vm', 'lp']
GENDERS = ['u']

# COMMAND ----------

# MAGIC %md # Stage Audio after converting to FLAC format
# MAGIC import subprocess
# MAGIC import sox
# MAGIC def stage_audio(statsFile, inputAudioDir, outputAudioDir, classList, makeLinks=True, outExt=None, outMaxAudioDuration=0):
# MAGIC   if not os.path.exists(outputAudioDir):
# MAGIC     os.makedirs(outputAudioDir)
# MAGIC
# MAGIC   if outExt is None or outExt == "":
# MAGIC     outExt = 'flac'
# MAGIC
# MAGIC   with open(statsFile, "r") as statsfp:
# MAGIC     for entry in statsfp:
# MAGIC       entries = entry.split()
# MAGIC       classId = int(entries[2])
# MAGIC       assert len(entries) >= 3
# MAGIC       assert classId in (0, 1)
# MAGIC
# MAGIC       clazz    = classList[classId]
# MAGIC       assert clazz in classList
# MAGIC       gender   = 'u'
# MAGIC
# MAGIC       inFilePattern = "%s/*%s*" % (inputAudioDir, entries[0])
# MAGIC       inFiles = glob.glob(inFilePattern)
# MAGIC       assert len(inFiles) > 0
# MAGIC       for inFile in inFiles:
# MAGIC         filename = os.path.basename(inFile)
# MAGIC         comp = filename.split(".")
# MAGIC         ext  = comp.pop()
# MAGIC         base = ".".join(comp)
# MAGIC         if makeLinks:
# MAGIC           #link outputAudioDir/basename.ext to inFile
# MAGIC           outFile = "%s/%s.%s" % (outputAudioDir, base, ext)
# MAGIC           os.symlink(inFile, outFile)
# MAGIC           print("Linked %s to %s" % (inFile, outFile))
# MAGIC         else:
# MAGIC           outFile = "%s/%s_%s_%s.%s" % (outputAudioDir, clazz, gender, base, outExt)
# MAGIC           tfm = sox.Transformer()
# MAGIC           if outMaxAudioDuration > 0:
# MAGIC             tfm.trim(0, outMaxAudioDuration)
# MAGIC           tfm.build_file(inFile, outFile)
# MAGIC           print("Converted %s to %s" % (inFile, outFile))
# MAGIC
# MAGIC # Convert original training MP3 audio to SEG_DURATION seconds of FLAC files
# MAGIC stage_audio("/dbfs/FileStore/AMD/lists/train.stats", ORIG_AUDIO_DIR, os.path.join(DATASET_DIST, 'train'), CLASSES, makeLinks=False, outExt='flac', outMaxAudioDuration=SEG_DURATION)
# MAGIC
# MAGIC # Convert original test MP3 audio to SEG_DURATION seconds of FLAC files
# MAGIC stage_audio("/dbfs/FileStore/AMD/lists/test.stats", ORIG_AUDIO_DIR, os.path.join(DATASET_DIST, 'test'), CLASSES, makeLinks=False, outExt='flac', outMaxAudioDuration=SEG_DURATION)

# COMMAND ----------

# MAGIC %md # Feature Extraction

# COMMAND ----------

from acoustic_cnn import features

# do filter-bank feature extraction on each test audio file
features.process_audio(os.path.join(DATASET_DIST, 'test'), os.path.join(FEATS_DIST, 'test'), debug=False, SeqLength=WIDTH, NumFeats=FB_HEIGHT)

# do filter-bank feature extraction on each training audio file
features.process_audio(os.path.join(DATASET_DIST, 'train'), os.path.join(FEATS_DIST, 'train'), debug=False, SeqLength=WIDTH, NumFeats=FB_HEIGHT)

# COMMAND ----------

# MAGIC %md # Generate Folds

# COMMAND ----------

from acoustic_cnn import folds

out_dir = os.path.join(EXPTS_INT, 'folds')
if os.path.exists(out_dir):
  shutil.rmtree(out_dir, ignore_errors=True)

# combine feature files into K fold files to allow K-fold X-validation

# for test files
folds.generate_folds(
    os.path.join(FEATS_DIST, 'test'),
    '.fb.npz',
    output_dir=os.path.join(EXPTS_INT, 'folds'),
    group='test',
    input_shape=(WIDTH, FB_HEIGHT),
    output_shape=(FB_HEIGHT, WIDTH, COLOR_DEPTH),
    num_folds=NUM_FOLDS
)

# and for train files
folds.generate_folds(
    os.path.join(FEATS_DIST, 'train'),
    '.fb.npz',
    output_dir=os.path.join(EXPTS_INT, 'folds'),
    group='train',
    input_shape=(WIDTH, FB_HEIGHT),
    output_shape=(FB_HEIGHT, WIDTH, COLOR_DEPTH),
    num_folds=NUM_FOLDS
)

# COMMAND ----------

# MAGIC %md # Training and K-fold Cross-Validation

# COMMAND ----------

from acoustic_cnn import cnnbc

NUM_EPOCHS = 20
BATCH_SIZE = 8
DO_K_FOLD_X_VALIDATION = False
THRESHOLD=0.5

modelName   = os.path.join(MODEL_DIST, 'model.h5')
foldsFolder = os.path.join(EXPTS_INT, 'folds')
input_shape = (FB_HEIGHT, WIDTH, COLOR_DEPTH)

cnnbc.train_and_validate(
  foldsFolder,
  input_shape,
  outModelFileName=modelName,
  numEpochs=NUM_EPOCHS,
  batchSize=BATCH_SIZE,
  doKfoldXValidation=DO_K_FOLD_X_VALIDATION,
  createModelRef=cnnbc.build_model,
  threshold=THRESHOLD)

if not os.path.exists(OUT_ARTIFACTS):
  os.makedirs(OUT_ARTIFACTS)

mfp = "%s/*.h5" % MODEL_DIST
modelFiles=glob.glob(mfp)
for file in modelFiles:
  shutil.copy(file, OUT_ARTIFACTS)

# COMMAND ----------

# MAGIC %md # Evaluation on held-out test set

# COMMAND ----------

THRESHOLD = 0.8
cnnbc.decode_and_evaluate(
  modelName,
  foldsFolder,
  'test',
  input_shape,
  THRESHOLD)


# COMMAND ----------

# MAGIC %md # Feature Extract, Decode, Evaluate and Get Timings

# COMMAND ----------

from acoustic_cnn import decoder, common
import time
from sklearn.metrics import classification_report

THRESHOLD = 0.8
label_binarizer, clazzes = common.build_label_binarizer()

class2id = dict()
for index, entry in enumerate(clazzes):
  class2id[entry] = index

pattern = "%s/test/*.flac" % DATASET_DIST
audioFiles = glob.glob(pattern)
reflabels = list()

modelName = os.path.join(OUT_ARTIFACTS, "model.h5.mva.h5")
start = time.time()
amdDec = decoder.CNNBC_Decoder(modelName)
print("Model loading took a total of %.2f secs" % (time.time() - start))

procStart = time.time()
totProcTime = 0
hyplabels = list()

numFiles = 0
for audio in audioFiles:
  clazz = common.get_filename(audio).split('_')[0]
  assert clazz in clazzes
  classid = label_binarizer.transform([clazz])[0][0]
  reflabels.append(classid)

  start = time.time()
  score = amdDec.process(audio)
  totProcTime += (time.time() - start)
  hyp = int(score>=THRESHOLD)
  hyplabels.append(hyp)
  numFiles += 1
  print("%s \t %d \t %.6f \t %d" % (audio, classid, score, hyp))

print("\nFeatex and Decode on %d files took a total of %.2f secs with an average of %.2fmsecs/file" % (numFiles, totProcTime, (totProcTime*1000/numFiles)))
print(classification_report(reflabels, hyplabels, target_names=clazzes))

# COMMAND ----------
