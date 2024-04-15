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

pp = pprint.PrettyPrinter(depth=2)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.face_custom_module import FaceCustomLitModule
from src.models.components.age_classifier import AgeClassifier
from src.models.components.gender_classifier import GenderClassifier
from src.models.components.race_classifier import RaceClassifier
from src.models.components.skintone_classifier import SkintoneClassifier
from src.models.components.emotion_classifier import EmotionClassifier
from src.models.components.masked_classifier import MaskedClassifier
from src.models.components.backbone_maker import BackboneMaker
from src.data.components.face_dataset import FaceDataset
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


class Predictor:
    def __init__(self, args):
        self.ckpt_dir = args.ckpt_dir
        self.root_dir = args.root_dir
        self.image_list = args.image_list
        self.image_path = args.image_path
        self.image_dir = args.image_dir
        self.csv_path = args.csv_path
        self.attributes = args.attributes.split("|")
        self.backbone_name = args.backbone_name
        self.backbone = BackboneMaker(
            name=self.backbone_name).to(device)
        if args.backbone_checkpoint:
            self.backbone_checkpoint = args.backbone_checkpoint
            self.state_dict = self.re_keys(
                torch.load(self.backbone_checkpoint))
        else:
            self.backbone_checkpoint = os.path.join(self.ckpt_dir,
                                                    "backbone",
                                                    "best.ckpt")
        self.race_head = RaceClassifier(num_of_features=512)
        self.gender_head = GenderClassifier(output_size=1, num_of_features=512)
        self.age_head = AgeClassifier()
        self.skintone_head = SkintoneClassifier()
        self.emotion_head = EmotionClassifier()
        self.masked_head = MaskedClassifier()
        for attr in self.attributes:
            setattr(self, "checkpoint" + attr, None)

    def re_keys(self, dict):
        new_dict = dict.copy()
        for key in dict.keys():
            new_key = "backbone." + key
            new_dict[new_key] = new_dict[key]
            del new_dict[key]
        return new_dict
    
    def load_model(self):
        model = {}
        model["backbone"] = self.backbone.eval()
        self.backbone.load_state_dict(
            self.state_dict)
        for attr in self.attributes:
            setattr(self, attr + "_checkpoint", os.path.join(
                self.ckpt_dir,
                attr,
                "last.ckpt"))
            getattr(self, attr + "_head").load_state_dict(
                torch.load(getattr(self, attr + "_checkpoint")))
            getattr(self, attr + "_head").to(device)
            getattr(self, attr + "_head").eval()
            model[attr] = getattr(self, attr + "_head")
        return model

    def model_predict_image(self, model,
                            image_path,
                            backbone_name="resnet50"):
        pred = {}
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
            representation = model["backbone"](img)
            for attr in self.attributes:
                if attr == "backbone":
                    continue
                pred[attr] = model[attr](representation)
        pred = self.posprocess_pred(pred)
        pred["file_name"] = image_name
        return pred

    def pred2labels(self):
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

    def torch2torchscript(self, input_path, output_path):
        model = FaceCustomLitModule()
        checkpoint = torch.load(input_path)
        model.load_state_dict(checkpoint["state_dict"])
        print("Model saved in {}".format(output_path))
        script = model.to_torchscript()
        torch.jit.save(script, output_path)
        return model

    def posprocess_pred(self, pred):
        race, gender, age, skintone, emotion, masked = self.pred2labels()
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
            if key != "age" and key != "gender":
                pred[key] = np.argmax(pred[key], axis=1)
                pred[key] = attrs_dict[key][str(pred[key][0])]
        if "age" in pred.keys():
            pred["age"] = np.round(pred["age"])
            pred["age"] = np.sum(pred["age"], axis=1).astype(int)
            pred["age"] = attrs_dict["age"][str(pred["age"][0])]
        if "gender" in pred.keys():
            pred["gender"] = np.round(pred["gender"]).astype(int)
            pred["gender"] = attrs_dict["gender"][str(pred["gender"])]
        return pred

    def model_predict_batchs(self, model, batchs):
        preds = {}
        for i, (input_tensor, img_name) in tqdm(enumerate(batchs)):
            pred = {}
            input_tensor = input_tensor.to(device)
            with torch.no_grad():
                representation = model["backbone"](input_tensor)
                for attr in self.attributes:
                    pred[attr] = model[attr](representation)
            pred = self.posprocess_pred(pred)
            pred["file_name"] = img_name[0]
            preds[img_name[0]] = pred
        return preds

    def model_predict_img_dir(self, model):
        preds = {}
        for img_name in tqdm(os.listdir(self.image_dir)):
            img_path = os.path.join(self.image_dir, img_name)
            pred = self.model_predict_image(model, img_path)
            preds[img_name] = pred
        return preds

    def load_image_batch_from_dir(self,
                                  mean,
                                  std,
                                  image_size):
        print("Loading images from {}".format(self.root_dir))
        test_dataset = FaceDataset(
            self.root_dir,
            self.image_list,
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

    def warmup(self, model):
        input_tensor = torch.randn(1, 3, 224, 224).to(device)
        for _ in range(3):
            with torch.no_grad():
                representation = model["backbone"](input_tensor)
                for attr in self.attributes:
                    model[attr](representation)
        print("Warmup done.")
        return model

    def dict2csv(self, preds):
        df = pd.DataFrame.from_dict(preds, orient="index")
        df.to_csv(self.csv_path, index=False)
        print("Predictions saved in {}".format(self.csv_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--image_list", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--attributes", type=str)
    parser.add_argument("--backbone_name", type=str)
    parser.add_argument("--backbone_checkpoint", type=str)
    args = parser.parse_args()
    predictor = Predictor(args)
    model = predictor.load_model()
    predictor.warmup(model)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    image_size = 224
    if predictor.image_path:
        pred = predictor.model_predict_image(model, predictor.image_path)
        pp.pprint(pred)
    elif predictor.image_dir:
        preds = predictor.model_predict_img_dir(model)
        predictor.dict2csv(preds)
    else:
        batchs = predictor.load_image_batch_from_dir(
            mean,
            std,
            image_size
            )
        preds = predictor.model_predict_batchs(model, batchs)
        predictor.dict2csv(preds)
        pp.pprint(preds)
