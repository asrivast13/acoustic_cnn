#!/bin/bash

apt-get update && apt install -y git unzip git-lfs ffmpeg sox libsoxr-dev libsox-fmt-all
/databricks/python3/bin/pip install --upgrade pip
/databricks/python3/bin/pip install tensorflow-gpu
/databricks/python3/bin/pip install numpy pandas pydub soundfile speechpy sklearn
git clone https://github.com/td-amit/acoustic_cnn.git /tmp/acoustic_cnn
export PYTHONPATH=$PYTHONPATH:/tmp/acoustic_cnn/

exit 0
