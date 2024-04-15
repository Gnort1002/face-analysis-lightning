import argparse
import numpy as np
import pandas as pd
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import Compose
import rootutils
import pprint

from tqdm import tqdm

pp = pprint.PrettyPrinter(depth=4)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.face_custom_module import FaceCustomLitModule
from src.models.components.age_classifier import AgeClassifier
from src.models.components.gender_classifier import GenderClassifier
from src.models.components.backbone_maker import BackboneMaker
from src.data.components.face_dataset import FaceDataset
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def pred2labels():
    race = {
        "0": "Caucasian",
        "1": "Mongoloid",
        "2": "Negroid",
    }
    gender = {"1": "Male", "0": "Female"}
    skintone = {
        "3": "mid-light",
        "1": "light",
        "2": "mid-dark",
        "0": "dark"
    }
    emotion = {
        "4": "Neutral",
        "3": "Happiness",
        "1": "Disgust",
        "0": "Anger",
        "6": "Surprise",
        "2": "Fear",
        "5": "Sadness",
    }
    masked = {"1": "unmasked", "0": "masked"}
    age = {
        "0": "Baby",
        "1": "Kid",
        "2": "Teenager",
        "3": "20-30s",
        "4": "40-50s",
        "5": "Senior",
    }
    return race, gender, age, skintone, emotion, masked


def torch2torchscript(input_path, output_path):
    model = FaceCustomLitModule()
    checkpoint = torch.load(input_path)
    model.load_state_dict(checkpoint["state_dict"])
    print("Model saved in {}".format(output_path))
    script = model.to_torchscript()
    torch.jit.save(script, output_path)
    return model


def load_model(ckpt_path):
    model = FaceCustomLitModule.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()
    return model


def posprocess_pred(pred):
    race, gender, age, skintone, emotion, masked = pred2labels()
    attrs_dict = {
        "race": race,
        "gender": gender,
        "age": age,
        "skintone": skintone,
        "emotion": emotion,
        "masked": masked,
    }
    for key in pred.keys():
        pred[key] = pred[key].cpu().numpy()
        pred[key] = np.argmax(pred[key], axis=1)
        pred[key] = attrs_dict[key][str(pred[key][0])]
        if key == "age":
            pred[key] = pred[key].cpu().numpy()
            pred[key] = np.round(pred[key])
            pred[key] = np.sum(pred[key], axis=1).astype(int)
            pred[key] = attrs_dict[key][str(pred[key][0])]
    return pred


def model_predict_batchs(model, batchs):
    model.freeze()
    preds = {}
    for i, (input_tensor, img_name) in tqdm(enumerate(batchs)):
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            pred = model(input_tensor)
        pred = posprocess_pred(pred)
        pred["file_name"] = img_name[0]
        preds[img_name[0]] = pred
    return preds


def model_predict_image(model, image_path, backbone_name="resnet50"):
    model.freeze()
    image_name = os.path.basename(image_path)
    img = Image.open(image_path)
    img = img.convert("RGB")
    if backbone_name == "inception_resnet_v1":
        transform = Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                fixed_image_standardization,
            ]
        )
    if backbone_name == "resnet50":
        transform = Compose(
            [
                transforms.Resize((240, 240)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        pred = model(img)
    pred = posprocess_pred(pred)
    pred["file_name"] = image_name
    return pred


def model_predict_img_dir(model, img_dir, backbone_name="resnet50"):
    model.freeze()
    preds = {}
    for img_name in tqdm(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        pred = model_predict_image(model,
                                   img_path,
                                   backbone_name=backbone_name)
        preds[img_name] = pred
    return preds


def load_image_batch_from_dir(root_dir, image_list, mean, std, image_size):
    print("Loading images from {}".format(root_dir))
    test_dataset = FaceDataset(
        root_dir,
        image_list,
        mean,
        std,
        image_size,
        crop_size=image_size,
        mode="predict",
        transform=True,
        predict_mode=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    return test_dataloader


def warmup(model):
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    model.freeze()
    for _ in range(3):
        with torch.no_grad():
            model(input_tensor)
    print("Warmup done.")
    return model


def dict2csv(preds, csv_path):
    df = pd.DataFrame.from_dict(preds, orient="index")
    df.to_csv(csv_path, index=False)
    print("Predictions saved in {}".format(csv_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--image_list", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--backbone_name", type=str)
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    root_dir = args.root_dir
    image_list = args.image_list
    image_path = args.image_path
    image_dir = args.image_dir
    csv_path = args.csv_path
    backbone_name = args.backbone_name
    model = load_model(ckpt_path)
    warmup(model)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    image_size = 224
    if image_path:
        pred = model_predict_image(model, image_path,
                                   backbone_name=backbone_name)
        pp.pprint(pred)
    elif image_dir:
        preds = model_predict_img_dir(model, image_dir,
                                      backbone_name=backbone_name)
        dict2csv(preds, csv_path)
    else:
        batchs = load_image_batch_from_dir(root_dir,
                                           image_list,
                                           mean, std,
                                           image_size
                                           )
        preds = model_predict_batchs(model, batchs)
        dict2csv(preds, csv_path)
        pp.pprint(preds)
