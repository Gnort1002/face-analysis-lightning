#!/bin/zsh
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/inference.py --ckpt_path "/media/gnort/HDD6/Study/face-analysis-lightning/logs/train/runs/2024-01-15_08-56-46/checkpoints/epoch_001.ckpt" \
                        --csv_path "outputs/answer_gender_ir50_1.csv" \
                        --root_dir "/media/gnort/HDD6/Study/face-analysis-lightning/dataset/face/cropped_faces" \
                        --image_list "/media/gnort/HDD6/Study/face-analysis-lightning/dataset/face/image_lists/face_test.txt" \
                        --image_dir "/media/gnort/HDD6/Study/face-analysis-lightning/dataset/public_test/cropped_faces" \
                        --backbone_name "inception_resnet_v1" \
                        # --image_path "/media/gnort/HDD6/Study/face-analysis-lightning/dataset/face/cropped_faces/1.jpg"
                        
