#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Editor:
# iwatake (2020/09/06)



import argparse
import numpy as np
import sys
import os
import glob
import shutil
import struct
from random import shuffle

try:
    from PIL import Image
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have Pillow installed.
For installation instructions, see:
http://pillow.readthedocs.io/en/stable/installation.html""".format(err))


height = 224
width = 224

parser = argparse.ArgumentParser()
parser.add_argument('--inDir', required=True, help='Input directory')
parser.add_argument('--outDir', required=True, help='Output directory')

args = parser.parse_args()
CALIBRATION_DATASET_LOC = args.inDir + '/*.jpg'


# images to test
img_file_list = []
print("Location of dataset = " + CALIBRATION_DATASET_LOC)
img_file_list = glob.glob(CALIBRATION_DATASET_LOC)
print("Total number of images = " + str(len(img_file_list)))

# output
outDir  = args.outDir

# prepare output
if not os.path.exists(outDir):
	os.makedirs(outDir)

# Convert and output images  (use \n (do not use \r\n))
with open(outDir + "/list.txt", "wb") as f:
	for img_file in img_file_list:
		img = Image.open(img_file)
		img_resize = img.resize((width, height))
		basename_without_ext = os.path.splitext(os.path.basename(img_file))[0]
		img_resize.save(outDir+"/" + basename_without_ext + ".ppm")
		f.write((basename_without_ext + "\n").encode('utf-8'))
