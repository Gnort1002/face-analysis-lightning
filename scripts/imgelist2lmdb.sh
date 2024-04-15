#!/bin/zsh
python  utils/imagelist2lmdb_face.py \
        -l dataset/face/image_lists/pixta_masked_train.txt \
        -s dataset/face/cropped_faces \
        -a masked \
        -d dataset/face/lmdb/masked