#!/bin/zsh
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/inference_2.py --ckpt_dir "/media/gnort/HDD6/Study/face-analysis-lightning/logs/train/runs/2024-01-15_IR50_balanced_gender_v2/checkpoints" \
                        --csv_path "outputs/answer_gender_ir50_1.csv" \
                        --root_dir "/media/gnort/HDD6/Study/face-analysis-lightning/dataset/face/cropped_faces" \
                        --image_list "/media/gnort/HDD6/Study/face-analysis-lightning/dataset/face/image_lists/face_test.txt" \
                        --image_dir "/media/gnort/HDD6/Study/face-analysis-lightning/dataset/public_test/cropped_faces" \
                        --attributes "gender" \
                        --backbone_name "inception_resnet_v1" \
                        --backbone_checkpoint "/media/gnort/HDD6/Study/face-analysis-lightning/weights/20180402-114759-vggface2.pt"
                        # --image_path "/media/gnort/HDD6/Study/face-analysis-lightning/dataset/face/cropped_faces/1.jpg"
                        
