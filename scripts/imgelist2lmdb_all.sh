#!/bin/zsh
python  utils/imagelist2lmdb_all_attrs.py \
        -l dataset/face/image_lists/face_age_train.txt \
        -s dataset/face/cropped_faces \
        -d dataset/face/lmdb/all