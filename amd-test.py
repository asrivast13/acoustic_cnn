# Databricks notebook source
import sys
import os
import shutil
import glob
import time
import numpy as np
from sklearn.metrics import classification_report
sys.path.append("/tmp/")
sys.path.append("/tmp/acoustic_cnn")
from acoustic_cnn import decoder, common

DATASET_DIST = '/dbfs/FileStore/AMD/input'
modelName    = '/dbfs/FileStore/AMD/expts2/model.h5'

THRESHOLD = 0.8
label_binarizer, clazzes = common.build_label_binarizer()

pattern = "%s/test/*.flac" % DATASET_DIST
audioFiles = glob.glob(pattern)
reflabels = list()

start = time.time()
amdDec = decoder.CNNBC_Decoder(modelName)
print("Model loading took a total of %.2f secs" % (time.time() - start))

procStart = time.time()
totProcTime = 0
hyplabels = list()
procTimes = list()

numFiles = 0
for audio in audioFiles:
  clazz = common.get_filename(audio).split('_')[0]
  reflabels.append(clazz)
  
  start = time.time()
  score = amdDec.process(audio)
  delta = (time.time() - start)
  totProcTime += delta
  procTimes.append(delta*1000)
  hyp = int(score>=THRESHOLD)
  hyplabels.append(hyp)
  numFiles += 1
  classid = label_binarizer.transform([clazz])[0][0]
  print("%s \t %d \t %.6f \t %d \t %.2fmsecs" % (audio, classid, score, hyp, delta*1000))
  
procTimes = np.array(procTimes)
print("\nFeatex and Decode on %d files took a total of %.2f secs with an average of %.2fmsecs/file" % (numFiles, totProcTime, (totProcTime*1000/numFiles)))
print("\nProcessing time Max: %.2f | Min: %.2f | Mean: %.2f | STD: %.2f" % (procTimes.max(), procTimes.min(), procTimes.mean(), procTimes.std()))
refid = label_binarizer.transform(reflabels)
print(classification_report(refid, hyplabels, target_names=clazzes))

# COMMAND ----------


