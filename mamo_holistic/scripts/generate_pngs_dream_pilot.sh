#!/bin/bash
#
# Digital Mammography DREAM Challenge
# Training inference method

TARGET_WIDTH=832
TARGET_HEIGHT=1152

TARGET_WIDTH=896
TARGET_HEIGHT=1120


IMAGES_DIRECTORY="/media/HD/mamo/dream_pilot"
EXAMS_METADATA_FILENAME=$IMAGES_DIRECTORY"/exams_metadata.tsv"
IMAGES_CROSSWALK_FILENAME=$IMAGES_DIRECTORY"images_crosswalk.tsv"
PREPROCESS_IMAGES_DIRECTORY="/home/alalbiol/Data/mamo/dream_pilot_png_"$TARGET_WIDTH"x"$TARGET_HEIGHT


if [ ! -d "$IMAGES_DIRECTORY" ]; then
    echo "Creating the directory $IMAGES_DIRECTORY"
    mkdir -p $PREPROCESS_IMAGES_DIRECTORY
fi

echo "Resizing and converting $(find $IMAGES_DIRECTORY -name "*.dcm" | wc -l) DICOM images to PNG format"
find $IMAGES_DIRECTORY/ -name "*.dcm" | parallel "convert {} -interpolate bilinear -resize ${TARGET_WIDTH}x${TARGET_HEIGHT}! $PREPROCESS_IMAGES_DIRECTORY/{/.}.png"
echo "PNG images have been successfully saved to $PREPROCESS_IMAGES_DIRECTORY/."
cp $IMAGES_DIRECTORY/exams_metadata.tsv $PREPROCESS_IMAGES_DIRECTORY/exams_metadata.tsv
cp $IMAGES_DIRECTORY/images_crosswalk.tsv $PREPROCESS_IMAGES_DIRECTORY/images_crosswalk.tsv

echo "Preprocessing Done"

