# %%
import os
import os.path as osp
import random

import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import face_alignment

sns.set_theme(style="white")

# %%
data_dir = "../../data/raw" 
img_dir = osp.join(data_dir, "cropped_faces")
img_fnames = os.listdir(img_dir)

# %%
img_fname = random.sample(img_fnames, 1)[0]
img_fpath = osp.join(img_dir, img_fname)
img = cv2.imread(img_fpath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

# %%
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device="cuda"
)

# %%
landmarks = fa.get_landmarks_from_image(
    img,
    detected_faces=[[0, 0, img.shape[1], img.shape[0]]],
    return_landmark_score=True,
)


# %%
