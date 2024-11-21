from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS
import base64
import argparse
import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
import cv2
import os
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
import colorsys
import pickle as pkl
import pandas as pd
import json

from models.resnet import resnet50, ResNet50_Weights
import models.densenet as densenet
from utils.dataloader_med import Augmentation, ChestX_ray14, ChestX_ray14_det, ChestX_ray14_bbox
matplotlib.use("Agg")  # 또는 다른 백엔드 선택

import sys
import shutil

# Add the LLaVA folder to the PYTHONPATH to access LLaVAModel class for NLE generation. Hamza
sys.path.append(os.path.join(os.path.dirname(__file__), 'LLaVA', 'llava', 'serve'))
print(sys.path)  
from llava_model import LLaVAModel
import requests
from werkzeug.utils import secure_filename

# for locking the process, to make sure only one request at a time, other will be queued. 
import threading
#from datastore_retrieval import get_retrieved_info_for_image


"""
update: added --base_dir arg to prevent multiple changes of path in the code. 
Updated by Hamza
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Set up paths with base directory")

    # Add the base directory argument
    parser.add_argument("--base_dir", default="/app", help="Base directory for all paths")

    # Use base_dir to set default paths for other arguments
    parser.add_argument("--version", default="iitp", help="iitp, ccrc")
    parser.add_argument("--pt_weight", default=None, help="Path to weight")
    parser.add_argument("--example_root", default=None, help="Path to D_probe")
    parser.add_argument("--heatmap_save_root", default=None, help="Path to saved img")
    parser.add_argument("--num_example", default=1, type=int, help="# of examples to be used")
    parser.add_argument("--util_root", default=None, help="Path to utils")
    parser.add_argument("--map_root", default=None, help="Path to map")
    parser.add_argument("--thrs", default=None, help="Path to threshold")
    parser.add_argument("--img_names", default=None, help="Path to img names")
    parser.add_argument("--uploaded_image",default=None, help="Path to uploaded image")

    args = parser.parse_args()

    # Use base_dir to construct other paths dynamically
    args.pt_weight = os.path.join(args.base_dir, "utils/densenet121_CXR_0.3M_mocov2.pth")
    #args.example_root = os.path.join(args.base_dir, "examples/medical_input/test_img")
    
    #++++++++++++++++++++++++++++++++++++
    # added by Hamza for generting concepts of uploaded image .JPG
    # this is where the dataloader needs to look for the image.
    args.example_root = os.path.join(args.base_dir, "uploaded_image")
    # this is where the image will be stored.
    args.uploaded_image = os.path.join(args.base_dir, "uploaded_image/image")
    #++++++++++++++++++++++++++++++++++++
    args.heatmap_save_root = os.path.join(args.base_dir, "heatmap/iitp_v2")
    args.util_root = os.path.join(args.base_dir, "utils")
    args.map_root = os.path.join(args.base_dir, "heatmap_info/med")
    args.thrs = os.path.join(args.base_dir, "utils/densenet121_thrs.csv")
    #args.img_names = ""
    #args.img_names = os.path.join(args.base_dir, "img_name.txt")

    return args

def show(img, **kwargs):
    img = np.array(img)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    img -= img.min()
    img /= img.max()
    plt.imshow(img, **kwargs)
    plt.axis("off")


def get_alpha_cmap(cmap):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    else:
        c = np.array((cmap[0] / 255.0, cmap[1] / 255.0, cmap[2] / 255.0))

        cmax = colorsys.rgb_to_hls(*c)
        cmax = np.array(cmax)
        cmax[-1] = 1.0

        cmax = np.clip(np.array(colorsys.hls_to_rgb(*cmax)), 0, 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [c, cmax])

    alpha_cmap = cmap(np.arange(256))
    alpha_cmap[:, -1] = np.linspace(0, 0.85, 256)
    alpha_cmap = ListedColormap(alpha_cmap)

    return alpha_cmap


def concept_attribution_maps(
    cmaps,
    args,
    model,
    example_loader,
    num_top_neuron=5,
    percentile=90,
    alpha=0.7,
    gt=False,
):

    if args.version == "iitp":
        # shap 
        shap_path = f"{args.util_root}/shap_nih_train_densenet121.pkl" # ccrc: f"{args.util_root}/RN50_ImageNet_class_shap.pkl"
        with open(shap_path, "rb") as f:
            shap_value = pkl.load(f)
        # concept 
        concept_path = f"{args.util_root}/mimic_nouns_40_base_tem_adp_95_penultimate.pkl" # ccrc: f"{args.util_root}/NM_img_val_80k_tem_adp_5_layer4.pkl"
        with open(concept_path, "rb") as f:
            l4_concept = pkl.load(f)
            # l4_concept = l4_concept[0]
        example_dir = '/app/images/med_dense_example_pen'
    elif args.version == "ccrc":
        # shap 
        shap_path = f"{args.util_root}/RN50_ImageNet_class_shap.pkl"
        with open(shap_path, "rb") as f:
            shap_value = pkl.load(f)
        # concept 
        concept_path = f"{args.util_root}/NM_img_val_80k_tem_adp_5_layer4.pkl"
        with open(concept_path, "rb") as f:
            l4_concept = pkl.load(f)

    c_heatmap = []
    s_heatmap = []
    cc_val = []
    sc_val = []
    sc_idx = []
    final_pred = None

    ### Class concept Atribute Maps ####
    for j, (img, label) in enumerate(tqdm(example_loader)): # pred 없는 경우 pass
        ### predict
        # img = img.squeeze()
        img = img.to(args.device)
        # predict = model(img.to(args.device))
        predict = model(img)
        predict = predict.sigmoid()[0].cpu().detach().numpy()

        pred_idx = np.where(predict >= args.threshold)[0]
        
        # get the probablity
        pred_prob = predict[pred_idx]
        pred_class = [args.class_name[i] for i in pred_idx]

        # predict = model(img)
        # predict = predict[0].cpu().detach().numpy()
        # predict = np.argmax(predict)

        os.makedirs(f"{args.heatmap_save_root}", exist_ok=True)
        if pred_class == []:
            continue
        else:
            feature_maps = model.extract_feature_map_4(img)
            
            feature_maps = feature_maps[0].cpu().detach().numpy()
            feature_maps = feature_maps.transpose(1, 2, 0)
            if gt:
                most_important_concepts = np.argsort(shap_value[label.item()])[::-1][
                    :num_top_neuron
                ]
            else:
                imp_concepts = []
                for i in pred_idx:
                    most_important_concepts = np.argsort(shap_value[i])[::-1][
                        :num_top_neuron
                    ]
                    imp_concepts.append(most_important_concepts)
            
            if imp_concepts is not None:
                for pred, most_important in enumerate(imp_concepts):
                    show(img.to('cpu')[0])
                    concepts = []

                    for i, c_id in enumerate(most_important):
                        cmap = cmaps[i]

                        concepts.append(l4_concept[0][c_id])
                        concepts.append(c_id)
                        if args.version == 'ccrc':
                            directory_contents = os.listdir(
                                "./images/example_val_l4_top2/" + str(f"{c_id:04d}")
                            )
                        elif args.version == 'iitp':
                            directory_contents = os.listdir(
                                f"{example_dir}/images/" + str(f"{c_id:04d}")
                            )
                        concepts.append(directory_contents)
                        # heatmap = feature_maps[:, :, c_id]

                        # sigma = np.percentile(feature_maps[:, :, c_id].flatten(), percentile)
                        # heatmap = heatmap * np.array(heatmap > sigma, np.float32)

                        # heatmap = cv2.resize(heatmap[:, :, None], (224, 224))
                        # show(heatmap, cmap=cmap, alpha=0.9)
                    # plt.show()

                    # if gt:
                    #     plt.savefig(
                    #         f"{args.heatmap_save_root}/{label.item():04d}/class_attribute_n{num_top_neuron}_p{percentile}_a90/Class_att_gt{label.item():04d}_{(j):02d}.jpg"
                    #     )
                    # else:
                    #     # args.pred_class = pred_class[pred]
                    #     plt.savefig(
                    #         f"{args.heatmap_save_root}/{img_name[0].split('.')[0]}_{pred_class[pred]}_Class_att.jpg",
                    #         bbox_inches="tight",
                    #         pad_inches=0,
                    #     )
                    # plt.clf()

    #### Class overall Atribute Maps ####
    for j, (img, label) in enumerate(tqdm(example_loader)):
        img = img.to(args.device)
        predict = model(img)
        predict = predict.sigmoid()[0].cpu().detach().numpy()
        pred_idx = np.where(predict >= args.threshold)[0]
        pred_class = [args.class_name[i] for i in pred_idx]

        os.makedirs(f"{args.heatmap_save_root}", exist_ok=True)
        if pred_class == []:
            continue
        else:
            # show(img[0])
            feature_maps = model.extract_feature_map_4(img)
            feature_maps = feature_maps[0].cpu().detach().numpy()
            feature_maps = feature_maps.transpose(1, 2, 0)

            if gt:
                most_important_concepts = np.argsort(shap_value[label.item()])[::-1][
                    :num_top_neuron
                ]
            else:
                imp_concepts = []
                for i in pred_idx:
                    most_important_concepts = np.argsort(shap_value[i])[::-1][
                        :num_top_neuron
                    ]
                    imp_concepts.append(most_important_concepts)
            if imp_concepts is not None:
                for pred, most_important in enumerate(imp_concepts):
                    show(img.to('cpu')[0])

                    overall_heatmap = np.zeros((224, 224))
                    temp_weight = []

                    for i, c_id in enumerate(most_important):
                        cmap = cmaps[i]
                        heatmap = feature_maps[:, :, c_id]

                        # sigma = np.percentile(feature_maps[:,:,c_id].flatten(), percentile)
                        # heatmap = heatmap * np.array(heatmap > sigma, np.float32)

                        heatmap = cv2.resize(heatmap[:, :, None], (224, 224))
                        if gt:
                            weight = shap_value[label.item()][c_id] / np.sum(
                                shap_value[label.item()][most_important]
                            )
                        else:
                            weight = shap_value[i][c_id] / np.sum(
                                shap_value[i][most_important]
                            )
                        overall_heatmap += heatmap * weight
                        temp_weight.append(weight)

                    c_heatmap.append(overall_heatmap)
                    cc_val.append(temp_weight)
                    show(overall_heatmap, cmap="Reds", alpha=0.5)

                    if gt:
                        plt.savefig(
                            f"{args.heatmap_save_root}/{label.item():04d}/class_overall_n{num_top_neuron}_p0_a50/Class_ovr_gt{label.item():04d}_{(j%args.num_example):02d}.jpg"
                        )
                    else:
                        final_pred = pred_class[pred]
                        final_prob = pred_prob[pred]
                        plt.savefig(
                            #print("in concept function: ",args.img_names)
                            #print("in concept function if access as list: ",args.img_names)
                            #print("in concept function if split with . and pick the first part: ",args.img_names.split('.'[0]))
                            f"{args.heatmap_save_root}/{args.img_names}_{pred_class[pred]}_Class_ovr.jpg",
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                    plt.clf()
                    plt.close()

    # with open(f"{args.map_root}/cc_val.pkl", "wb") as f:
    #     pkl.dump(cc_val, f)
    # cc_val = None

    # with open(f"{args.map_root}/c_heatmap.pkl", "wb") as f:
    #     pkl.dump(c_heatmap, f)
    # c_heatmap = None

    #### sample concept Atribute Maps ####
    # for j, (img, label) in enumerate(tqdm(example_loader)):
    #     img = img.to(args.device)
    #     predict = model(img)
    #     predict = predict.sigmoid()[0].cpu().detach().numpy()
    #     pred_idx = np.where(predict >= args.threshold)[0]
    #     pred_class = [args.class_name[i] for i in pred_idx]

    #     os.makedirs(f"{args.heatmap_save_root}", exist_ok=True)
    #     if pred_class == []:
    #         continue
    #     else:
    #         # show(img[0])
    #         feature_maps = model.extract_feature_map_4(img)
    #         imp_concepts = []
    #         for c_i, i in enumerate(pred_idx):
    #             show(img.to('cpu')[0])

    #             sample_shap = model._compute_taylor_scores(img, i)
    #             sample_shap = sample_shap[0][0][0, :, 0, 0]
    #             sample_shap = sample_shap.cpu().detach().numpy()
    #             if type(feature_maps[0])==np.ndarray:
    #                 feature_maps = feature_maps
    #             else:
    #                 feature_maps = feature_maps[0].cpu().detach().numpy()
    #                 feature_maps = feature_maps.transpose(1, 2, 0)
    #             most_important_concepts = np.argsort(sample_shap)[::-1][:num_top_neuron]
    #             sc_idx.append(most_important_concepts)

    #             for i, c_id in enumerate(most_important_concepts):
    #                 cmap = cmaps[i]
    #                 heatmap = feature_maps[:, :, c_id]

    #                 sigma = np.percentile(feature_maps[:, :, c_id].flatten(), percentile)
    #                 concepts.append(l4_concept[0][c_id])
    #                 concepts.append(c_id)

    #                 if args.version == 'ccrc':
    #                     directory_contents = os.listdir(
    #                         "./images/example_val_l4_top2/" + str(f"{c_id:04d}")
    #                     )
    #                 elif args.version == 'iitp':
    #                     directory_contents = os.listdir(
    #                         f"{example_dir}/images/" + str(f"{c_id:04d}")
    #                     )

    #                 concepts.append(directory_contents)
                    # heatmap = heatmap * np.array(heatmap > sigma, np.float32)

                    # heatmap = cv2.resize(heatmap[:, :, None], (224, 224))
                    # show(heatmap, cmap=cmap, alpha=0.9)

                # plt.savefig(
                #     f"{args.heatmap_save_root}/{img_name[0].split('.')[0]}_{pred_class[c_i]}_sample_att.jpg",
                #     bbox_inches="tight",
                #     pad_inches=0,
                # )
                # plt.clf()

    # with open(f"{args.map_root}/sc_idx.pkl", "wb") as f:
    #     pkl.dump(sc_idx, f)
    # sc_idx = None

    #### Sample overall Atribute Maps ####
    # for j, (img, label) in enumerate(tqdm(example_loader)):
    #     img = img.to(args.device)
    #     predict = model(img)
    #     predict = predict.sigmoid()[0].cpu().detach().numpy()
    #     pred_idx = np.where(predict >= args.threshold)[0]
    #     pred_class = [args.class_name[i] for i in pred_idx]

    #     os.makedirs(f"{args.heatmap_save_root}", exist_ok=True)
    #     if pred_class == []:
    #         continue
    #     else:
    #         # show(img[0])
    #         feature_maps = model.extract_feature_map_4(img)
    #         for c_i, i in enumerate(pred_idx):
    #             show(img.to('cpu')[0])
    #             sample_shap = model._compute_taylor_scores(img, i)
    #             sample_shap = sample_shap[0][0][0, :, 0, 0]
    #             sample_shap = sample_shap.cpu().detach().numpy()
    #             if type(feature_maps[0])==np.ndarray:
    #                 feature_maps = feature_maps
    #             else:
    #                 feature_maps = feature_maps[0].cpu().detach().numpy()
    #                 feature_maps = feature_maps.transpose(1, 2, 0)
    #             most_important_concepts = np.argsort(sample_shap)[::-1][:num_top_neuron]
    #             overall_heatmap = np.zeros((224, 224))
    #             if args.version == "ccrc":
    #                 with open(
    #                     "./utils/imagenet_labels.txt", "r"
    #                 ) as f:  # directory of imagenet_labels.txt
    #                     words = (f.read()).split("\n")
    #             elif args.version == "iitp":
    #                 with open('/app/utils/nih_labels.txt', 'r') as f:
    #                     words = (f.read()).split("\n")
    #             concepts.append([words[i]])
    #             temp_weight = []
                # for i, c_id in enumerate(most_important_concepts):
                #     cmap = cmaps[i]
                #     heatmap = feature_maps[:, :, c_id]

                #     # sigma = np.percentile(feature_maps[:,:,c_id].flatten(), percentile)
                #     # heatmap = heatmap * np.array(heatmap > sigma, np.float32)

                #     heatmap = cv2.resize(heatmap[:, :, None], (224, 224))
                #     weight = sample_shap[c_id] / np.sum(sample_shap[most_important_concepts])
                #     overall_heatmap += heatmap * weight
                #     temp_weight.append(weight)

                # sc_val.append(temp_weight)
                # show(overall_heatmap, cmap="Reds", alpha=0.5)

                # plt.savefig(
                #     f"{args.heatmap_save_root}/{img_name[0].split('.')[0]}_{pred_class[c_i]}_sample_ovr.jpg",
                #     bbox_inches="tight",
                #     pad_inches=0,
                # )
                # plt.clf()

    # with open(f"{args.map_root}/sc_val.pkl", "wb") as f:
    #     pkl.dump(sc_val, f)
    # sc_val = None
    # with open(f"{args.map_root}/s_heatmap.pkl", "wb") as f:
    #     pkl.dump(s_heatmap, f)
    # s_heatmap = None
    if final_pred == None:
        final_pred = None
        final_prob = 0
        concepts = "Ther are no concepts."
    else:
        final_pred = pred_class[pred]
    return concepts, final_pred, final_prob

def load_threshold(thrs_path):
    Eval = pd.read_csv(thrs_path)
    thrs = [Eval["bestthr"][Eval[Eval["label"] == "Atelectasis"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Cardiomegaly"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Effusion"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Infiltration"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Mass"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Nodule"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Pneumonia"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Pneumothorax"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Consolidation"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Edema"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Emphysema"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Fibrosis"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Pleural thickening"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Hernia"].index[0]]]
    return thrs

def infer(args):
    # here initalization of args is creating the filename issue,
    #args = parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## Load model ##
    if args.version == "ccrc":
        ##### ResNET50 #####
        
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.eval()
        featdim = 2048

        transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        examples = tv.datasets.ImageFolder(args.example_root, transform=transform)
        example_loader = torch.utils.data.DataLoader(examples, batch_size=1, shuffle=False)

    elif args.version == "iitp":
        args.threshold= load_threshold(args.thrs)
        with open('/app/utils/nih_labels.txt', 'r') as f: 
            args.class_name = (f.read()).split('\n')
        ##### DenseNET121 #####
        checkpoint = torch.load(args.pt_weight, map_location='cpu')
        model = densenet.__dict__['densenet121'](num_classes=14)
        # if layer == 'penultimate':
        #     model.classifier = torch.nn.Identity()

        if 'state_dict' in checkpoint.keys():
            checkpoint_model = checkpoint['state_dict']
        elif 'model' in checkpoint.keys():
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f'Model weigth load : {msg}')
        model.to(args.device)
        model.eval()
        featdim = 1024

        transform = Augmentation(normalize="chestx-ray").get_augmentation("full_224", "val")
        no_normalize = Augmentation(normalize="none").get_augmentation("full_224", "val")

        ### Read img names file
        # with open(args.img_names, 'r') as f:
        #     img_names = f.readlines()
        # img_names = [x.strip() for x in img_names]

        # split_path = '/app/utils/ChestX_Det_test.json'
        
        #args = parse_args()
        #base_dir = args.base_dir

        #split_path = os.path.join(base_dir, "utils/ChestX_Det_test.json")
        # split_path = '/home/ameer/iitp/ccrc-demo/backend/utils/ChestX_Det_test_one.json'
        # examples = ChestX_ray14_det(args.example_root, split_path, augment=transform, no_normalize_aug=no_normalize, num_class=14, target_img=img_names)

        # sampler = torch.utils.data.SequentialSampler(examples)
        # example_loader = torch.utils.data.DataLoader(
        #     examples, sampler=sampler,
        #     batch_size=1, #args.batch_size,
        #     num_workers=4, #args.num_workers,
        #     pin_memory=True, #args.pin_mem,
        #     drop_last=False,
        #     shuffle=False
        # )
        transform = tv.transforms.Compose([
                tv.transforms.Resize(256),
                tv.transforms.CenterCrop(224),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        examples = tv.datasets.ImageFolder(args.example_root, transform=transform)
        example_loader = torch.utils.data.DataLoader(examples, batch_size=1, shuffle=False)

    os.makedirs(f"{args.map_root}", exist_ok=True)

    cmaps = [
        get_alpha_cmap((54, 197, 240)),  ##blue
        get_alpha_cmap((210, 40, 95)),  ##red
        get_alpha_cmap((236, 178, 46)),  ##yellow
        get_alpha_cmap((15, 157, 88)),  ##green
        get_alpha_cmap((84, 25, 85)),  ##purple
        get_alpha_cmap((255, 0, 0)),  ##real red
    ]

    concepts, final_pred, final_prob = concept_attribution_maps(
        cmaps,
        args,
        model,
        example_loader,
        num_top_neuron=3,
        percentile=70,
        alpha=0.8,
        gt=False,
    )

    # # get img_label from examples
    # img_label = examples.get_img_label()

    # # get all the img_labels and assign them to the predictions list
    # predictions = []
    # for i in range(len(img_label[0])):
    #     predictions.append(img_label[0][i])

    return final_pred , concepts, final_prob


app = Flask(__name__)
# added by Hamza
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status="healthy"), 200



UPLOAD_FOLDER = './examples/0'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# to handle out of memory CUDA error, load the both models once

# Pass the image and prompt to the LLaVA model
model_path = "/app/LLaVA/llava-lora-llama-kg-model-nle"
# Initialize the model class
llava_model = LLaVAModel(model_path=model_path, load_4bit=True, device='cuda')
# Global lock
request_lock = threading.Lock()


@app.route("/explain", methods=["POST"])
def explain():

    with request_lock:
        # clear the torch cache
        torch.cuda.empty_cache()
        
        response_data = {}
        if 'image' not in request.files:
            return jsonify({"error": "No image file found"}), 400
        file = request.files['image']
        # Ensure the file is a .jpg file
        if file.filename == '' or not file.filename.lower().endswith('.jpg'):
            return jsonify({"error": "Only .jpg files are allowed."}), 400
        
        # args.img_names = file.filename

        # base_path = os.path.dirname(__file__)

        # img_name_path = os.path.join(args.base_dir, 'img_name.txt')

        
        # if not os.path.exists(img_name_path):
        #     with open(img_name_path, "w") as f:
        #         f.write(file.filename)
        # else:        
        #     with open(img_name_path, "w") as f:
        #         f.write(file.filename)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # added by Hamza

        args = parse_args()
        # Define the path to the 'uploaded_image' directory
        upload_directory = args.uploaded_image

        # Clear the directory to ensure it only contains the latest image
        for existing_file in os.listdir(upload_directory):
            file_path = os.path.join(upload_directory, existing_file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Save the uploaded image in the 'uploaded_image' directory
        filename = secure_filename(file.filename)
        saved_image_path = os.path.join(upload_directory, filename)
        file.save(os.path.join(upload_directory, filename))

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
        #print("default args.img_names vallue:", args.img_names)
        args.img_names = os.path.splitext(filename)[0]
        #print("args.img_names: ", args.img_names)
        #print(args.img_names)
        
        # infer() function will now return the predictions and the total number of predictions
        final_pred, concepts, final_prob = infer(args)

        # if final_pred, concepts and final_prob is none or empty then skip the NLE part, else run.

        if final_pred == None:
            final_pred = "No abnormality is present."
            str_final_prob = "0"
            concept_list = ["None","None"]
            reports = "There is no abnormality present."
            exp_pc_1 = f"http://{request.host}/images/default-image.jpg"
            
        else:
            # run the NLE model.
             #print("+++++++++++++++++++++PREDICTIONS+++++++++++++++++++++++++++++++++",final_pred)

            # from here pass the prediction and image to the generateNLE() and get NLEs for current image.
            # image for NLE generation
            nle = generateNLE(final_pred,saved_image_path)
            
            # concepts code starts here 
            # --------------------------
            concepts_str = ""
            concept_list = []
            concept_number = 1

            # Iterate through the concepts list
            for concept in concepts:
                # break after 2 concepts
                if concept_number > 2:
                    break
                # Check if the element is a list and its contents are not image filenames
                if isinstance(concept, list) and all(not str(item).endswith('.jpg') for item in concept):
                    # Join the list elements into a string
                    concept_list.append(concept)
                    joined_concept = ", ".join(concept)
                    # Append the concept number and the joined list to the final string
                    concepts_str += f"concept {concept_number}: {joined_concept} \n\n"
                    concept_number += 1
            
            # concepts code ends here 
            # --------------------------
            #print(concepts_str)

            # temp
            reports = nle

            #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            #print(final_prob)
            #print(type(final_prob))

            round_final_prob = round(float(final_prob), 2)

            # convert round_final_prob into string 

            str_final_prob = str(round_final_prob)
            #exp_pc_1 = f"http://{request.host}/images/{file.filename.split('.')[0]}_{final_pred}_Class_ovr.jpg"
            exp_pc_1 = f"http://{request.host.split(':')[0]}:9999/images/{file.filename.split('.')[0]}_{final_pred}_Class_ovr.jpg"

            

        response_data = {
            "pred": final_pred,
            "report": reports,
            #"exp-pc-1": f"http://127.0.0.1:5000/images/{file.filename.split('.')[0]}_{final_pred}_Class_ovr.jpg",
            #"exp-pc-1": f"http://{request.host}/images/{file.filename.split('.')[0]}_{final_pred}_Class_ovr.jpg",
            "exp-pc-1": exp_pc_1,
            # You may need to add the other image URL here too if needed
            #"input_image": f"http://127.0.0.1:5000/input_img/{file.filename}",
            "concept1": concept_list[0],
            "concept2": concept_list[1],
            "final_prob": str_final_prob

        }
    return jsonify(response_data)

from flask import send_from_directory

# Assuming your images are stored in a directory named 'heatmap'

@app.route('/images/<path:filename>')
def serve_image(filename):
    args = parse_args()
    base_path = os.path.join(args.base_dir, "heatmap/iitp_v2")
    return send_from_directory(base_path, filename)

@app.route('/input_img/<path:filename>')
def serve_input_image(filename):
    args = parse_args()
    base_path = os.path.join(args.base_dir, "examples/medical_input/test_box")
    return send_from_directory(base_path, filename)


# clean the nle 
def clean_nle_output(nle_output):
    # Remove the <s> and </s> tags from the text
    cleaned_output = nle_output.replace("<s>", "").replace("</s>", "")
    return cleaned_output



# Prompt construction function
def createPrompt(retrieved_info, final_pred):
    # Only use the first prediction from the predictions list
    import random
    seed_value = 42
    # Set the random seed for reproducibility
    random.seed(seed_value)

    # Question templates (defined previously in your reference code)
    question_templates = [
        "Which signs show that the patient has {pathologies}?",
        "Explain why these {pathologies} are present in the image.",
        "What evidence in the image indicates {pathologies}?",
        "How can you tell that the patient has {pathologies} from the image?",
        "What features suggest the presence of {pathologies} in this image?"
    ]
    first_prediction = final_pred
    
    # Combine the retrieved information into a string
    retrieved_info_str = "; ".join(retrieved_info)
    
    # Select a random question template
    question_template = random.choice(question_templates)
    
    # Construct the prompt using the retrieved information and the first prediction
    prompt = (
        f"The image-specific triplets from the knowledge graph are: {retrieved_info_str}. "
        f"And for the given image, {question_template.format(pathologies=first_prediction)}"
    )
    
    return prompt


# NLE generation function:
def generateNLE(final_pred, image_path):
    nle = None

    # Pass the image path to MedCLIP Docker container
    # Save the uploaded file to a temporary path if necessary
    temp_dir = "/tmp"  # This can be adjusted to any directory with write permissions
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Copy the image to a temporary directory (if needed)
    filename = os.path.basename(image_path)  # Extract the image name
    temp_image_path = os.path.join(temp_dir, filename)

    # Copy the image to the temporary directory if it is different
    if image_path != temp_image_path:
        shutil.copyfile(image_path, temp_image_path)

    # Pass the image to the MedCLIP Docker container
    url = "http://121.134.238.177:8080/predict"  # Replace with the appropriate host if running remotely

    # Open the saved image file and send it to the Docker container
    with open(temp_image_path, "rb") as image_file:
        files = {"image": image_file}
        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                retrieved_info = response.json()
            else:
                print(f"Error: {response.json().get('error', 'Unknown error')}")
                retrieved_info = None
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Docker container: {str(e)}")
            retrieved_info = None

    # Handle missing or unexpected retrieved_info
    if not retrieved_info or not isinstance(retrieved_info, list):
        retrieved_info = ["No relevant information found."]

    # Generate a prompt based on the prediction and retrieved information
    prompt = createPrompt(retrieved_info, final_pred)

    
    # Generate the response
    nle = llava_model.generate_response(image_file=temp_image_path, prompt=prompt)
    
    # Clean the NLE output
    cleaned_nle = clean_nle_output(nle)
    return cleaned_nle

# def generateNLE(final_pred,file):
#     nle = None

#     # Pass the file to MedCLIP Docker container
# image_file = "/home/ameer/iitp/ccrc-demo/backend/examples/medical_input/test_img/36212.png"
#     image_file = file_path
#     # Initialize the model class
#     llava_model = LLaVAModel(model_path=model_path, load_4bit=True, device='cuda')
#     # Generate the response
#     nle = llava_model.generate_response(image_file=image_file, prompt=prompt)
#     #print(nle)
#     cleaned_nle = clean_nle_output(nle)
#     return cleaned_nle#     # Save the uploaded file to a temporary path
#     temp_dir = "/tmp"  # This can be adjusted to any directory with write permissions
#     if not os.path.exists(temp_dir):
#         os.makedirs(temp_dir)

#     # Generate a secure filename and save the file
#     filename = secure_filename(file.filename)
#     file_path = os.path.join(temp_dir, filename)
#     file.save(file_path)

#     # Pass the file to MedCLIP Docker container
#     url = "http://121.134.238.177:8080/predict"  # Replace with the appropriate host if running remotely

#     # Open the saved image file and send it to the Docker container
#     with open(file_path, "rb") as image_file:
#         files = {"image": image_file}
#         try:
#             response = requests.post(url, files=files)
#             if response.status_code == 200:
#                 retrieved_info = response.json()
#             else:
#                 print(f"Error: {response.json().get('error', 'Unknown error')}")
#                 retrieved_info = None
#         except requests.exceptions.RequestException as e:
#             print(f"Error communicating with Docker container: {str(e)}")
#             retrieved_info = None
#     # get the retrived information for the given input image which is in file

#     #print(retrieved_info)
    
#     # generate a prompt based on the prediction and reterived information
#     prompt = createPrompt(retrieved_info,final_pred)
#     #print(prompt)


#     # pass the image, prompt to the LLaVA model
#     model_path = "/app/LLaVA/llava-lora-llama-kg-model-nle"
#     #

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(debug = True, host="0.0.0.0", port=8000)
    #app.run(host="0.0.0.0", port=443, ssl_context=("/etc/letsencrypt/live/your-domain.com/fullchain.pem", "/etc/letsencrypt/live/your-domain.com/privkey.pem"))
    #infer()
    #pass