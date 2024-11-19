#!/bin/bash
#
# Digital Mammography DREAM Challenge
# Training inference method

source /metadata/link_files.sh
cd /

nvidia-smi
python -V
pip show tensorflow
ls -la .

IMAGES_DIRECTORY="/media/HD/Data/mamo/dream_pilot"
EXAMS_METADATA_FILENAME="/media/HD/Data/mamo/dream_pilot/exams_metadata.tsv"
IMAGES_CROSSWALK_FILENAME="/media/HD/Data/mamo/dream_pilot/images_crosswalk.tsv"
PREPROCESS_IMAGES_DIRECTORY="/home/alalbiol/Data/mamo/dream_pilot_png_832x1152"


if [ ! -d "$IMAGES_DIRECTORY" ]; then
    echo "Creating the directory $IMAGES_DIRECTORY"
    mkdir -p $PREPROCESS_IMAGES_DIRECTORY
fi

echo "Resizing and converting $(find $IMAGES_DIRECTORY -name "*.dcm" | wc -l) DICOM images to PNG format"
find $IMAGES_DIRECTORY/ -name "*.dcm" | parallel "convert {} -interpolate bilinear -resize 832x1152! $PREPROCESS_IMAGES_DIRECTORY/{/.}.png" # faster than mogrify
echo "PNG images have been successfully saved to $PREPROCESS_IMAGES_DIRECTORY/."

echo "Preprocessing Done"

