#!/bin/bash

apt-get update && apt install -y git unzip git-lfs ffmpeg sox libsoxr-dev libsox-fmt-all
/databricks/python3/bin/pip install --upgrade pip
/databricks/python3/bin/pip install tensorflow
/databricks/python3/bin/pip install numpy pandas pydub soundfile speechpy sklearn
git clone https://github.com/td-amit/acoustic_cnn.git
export PYTHONPATH=$PYTHONPATH:/databricks/driver/acoustic_cnn/

exit 0
