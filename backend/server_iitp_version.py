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

from models.resnet import resnet50, ResNet50_Weights
import models.densenet as densenet
from utils.dataloader_med import Augmentation, ChestX_ray14, ChestX_ray14_det, ChestX_ray14_bbox
matplotlib.use("Agg")  # 또는 다른 백엔드 선택

def parse_args():
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("--version", default="iitp", help="iitp, ccrc")
    parser.add_argument("--pt_weight", default="/data/hyeonbae/project/ccrc-demo/backend/utils/densenet121_CXR_0.3M_mocov2.pth", help="Path to weight")
    
    # parser.add_argument("--example_root", default="/data/hyeonbae/project/ccrc-demo/backend/examples", help="Path to D_probe")
    # parser.add_argument("--example_root", default="/data/hyeonbae/project/ccrc-demo/backend/examples/medical_input/test_img", help="Path to D_probe")
    parser.add_argument("--example_root", default="/data/hyeonbae/project/ccrc-demo/backend/examples/medical_input/one_image", help="Path to D_probe")
    parser.add_argument(
        "--heatmap_save_root", default="/data/hyeonbae/project/ccrc-demo/backend/heatmap/med", help="Path to saved img"
    )
    parser.add_argument(
        "--num_example", default=1, type=int, help="# of examples to be used"
    )
    parser.add_argument("--util_root", default="/data/hyeonbae/project/ccrc-demo/backend/utils", help="Path to utils")
    parser.add_argument("--map_root", default="/data/hyeonbae/project/ccrc-demo/backend/heatmap_info/med", help="Path to utils")

    # version IITP
    parser.add_argument("--thrs", default="/data/hyeonbae/project/medical-neuron_concept/dataset/class_threshold/densenet121_thrs.csv", type=str, help="Path to threshold")
    return parser.parse_args()


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
        example_dir = '/data/hyeonbae/project/ccrc-demo/backend/images/med_dense_example_pen'
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

    #### Class concept Atribute Maps ####
    for j, (img, img_name, label) in enumerate(tqdm(example_loader)): # pred 없는 경우 pass
        ### predict
        # img = img.squeeze()
        img = img.to(args.device)
        # predict = model(img.to(args.device))
        predict = model(img)
        predict = predict.sigmoid()[0].cpu().detach().numpy()
        pred_idx = np.where(predict >= args.threshold)[0]
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
                        heatmap = feature_maps[:, :, c_id]

                        sigma = np.percentile(feature_maps[:, :, c_id].flatten(), percentile)
                        heatmap = heatmap * np.array(heatmap > sigma, np.float32)

                        heatmap = cv2.resize(heatmap[:, :, None], (224, 224))
                        show(heatmap, cmap=cmap, alpha=0.9)
                    # plt.show()

                    if gt:
                        plt.savefig(
                            f"{args.heatmap_save_root}/{label.item():04d}/class_attribute_n{num_top_neuron}_p{percentile}_a90/Class_att_gt{label.item():04d}_{(j):02d}.jpg"
                        )
                    else:
                        plt.savefig(
                            f"{args.heatmap_save_root}/{img_name[0].split('.')[0]}_{pred_class[pred]}_Class_att.jpg",
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                    plt.clf()

    #### Class overall Atribute Maps ####
    for j, (img, img_name, label) in enumerate(tqdm(example_loader)):
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
                        plt.savefig(
                            f"{args.heatmap_save_root}/{img_name[0].split('.')[0]}_{pred_class[pred]}_Class_ovr.jpg",
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                    plt.clf()
                    # plt.close()

    with open(f"{args.map_root}/cc_val.pkl", "wb") as f:
        pkl.dump(cc_val, f)
    cc_val = None

    with open(f"{args.map_root}/c_heatmap.pkl", "wb") as f:
        pkl.dump(c_heatmap, f)
    c_heatmap = None

    #### sample concept Atribute Maps ####
    for j, (img, img_name, label) in enumerate(tqdm(example_loader)):
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
            imp_concepts = []
            for i in pred_idx:
                show(img.to('cpu')[0])

                sample_shap = model._compute_taylor_scores(img, i)
                sample_shap = sample_shap[0][0][0, :, 0, 0]
                sample_shap = sample_shap.cpu().detach().numpy()
                feature_maps = feature_maps[0].cpu().detach().numpy()
                feature_maps = feature_maps.transpose(1, 2, 0)
                most_important_concepts = np.argsort(sample_shap)[::-1][:num_top_neuron]
                sc_idx.append(most_important_concepts)

                for i, c_id in enumerate(most_important_concepts):
                    cmap = cmaps[i]
                    heatmap = feature_maps[:, :, c_id]

                    sigma = np.percentile(feature_maps[:, :, c_id].flatten(), percentile)
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
                    heatmap = heatmap * np.array(heatmap > sigma, np.float32)

                    heatmap = cv2.resize(heatmap[:, :, None], (224, 224))
                    show(heatmap, cmap=cmap, alpha=0.9)

                plt.savefig(
                    f"{args.heatmap_save_root}/{img_name[0].split('.')[0]}_{pred_class[pred]}_sample_att.jpg",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.clf()

    with open(f"{args.map_root}/sc_idx.pkl", "wb") as f:
        pkl.dump(sc_idx, f)
    sc_idx = None

    #### Sample overall Atribute Maps ####
    for j, (img, img_name, label) in enumerate(tqdm(example_loader)):
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
            for i in pred_idx:
                show(img.to('cpu')[0])
                sample_shap = model._compute_taylor_scores(img, i)
                sample_shap = sample_shap[0][0][0, :, 0, 0]
                sample_shap = sample_shap.cpu().detach().numpy()
                feature_maps = feature_maps[0].cpu().detach().numpy()
                feature_maps = feature_maps.transpose(1, 2, 0)
                most_important_concepts = np.argsort(sample_shap)[::-1][:num_top_neuron]
                overall_heatmap = np.zeros((224, 224))
                if args.version == "ccrc":
                    with open(
                        "./utils/imagenet_labels.txt", "r"
                    ) as f:  # directory of imagenet_labels.txt
                        words = (f.read()).split("\n")
                elif args.version == "iitp":
                    with open('/data/hyeonbae/project/ccrc-demo/backend/utils/nih_labels.txt', 'r') as f:
                        words = (f.read()).split("\n")
                concepts.append([words[i]])
                temp_weight = []
                for i, c_id in enumerate(most_important_concepts):
                    cmap = cmaps[i]
                    heatmap = feature_maps[:, :, c_id]

                    # sigma = np.percentile(feature_maps[:,:,c_id].flatten(), percentile)
                    # heatmap = heatmap * np.array(heatmap > sigma, np.float32)

                    heatmap = cv2.resize(heatmap[:, :, None], (224, 224))
                    weight = sample_shap[c_id] / np.sum(sample_shap[most_important_concepts])
                    overall_heatmap += heatmap * weight
                    temp_weight.append(weight)

                sc_val.append(temp_weight)
                show(overall_heatmap, cmap="Reds", alpha=0.5)

                plt.savefig(
                    f"{args.heatmap_save_root}/{img_name[0].split('.')[0]}_{pred_class[pred]}_sample_ovr.jpg",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.clf()

    with open(f"{args.map_root}/sc_val.pkl", "wb") as f:
        pkl.dump(sc_val, f)
    sc_val = None
    with open(f"{args.map_root}/s_heatmap.pkl", "wb") as f:
        pkl.dump(s_heatmap, f)
    s_heatmap = None

    return concepts

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

def infer():
    args = parse_args()
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
        with open('/data/hyeonbae/project/ccrc-demo/backend/utils/nih_labels.txt', 'r') as f: 
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

        # split_path = '/data/hyeonbae/project/ccrc-demo/backend/utils/ChestX_Det_test.json'
        split_path = '/data/hyeonbae/project/ccrc-demo/backend/utils/ChestX_Det_test_one.json'
        examples = ChestX_ray14_det(args.example_root, split_path, augment=transform, no_normalize_aug=no_normalize, num_class=14)

        sampler = torch.utils.data.SequentialSampler(examples)
        example_loader = torch.utils.data.DataLoader(
            examples, sampler=sampler,
            batch_size=1, #args.batch_size,
            num_workers=4, #args.num_workers,
            pin_memory=True, #args.pin_mem,
            drop_last=False,
            shuffle=False
        )

    os.makedirs(f"{args.map_root}", exist_ok=True)

    cmaps = [
        get_alpha_cmap((54, 197, 240)),  ##blue
        get_alpha_cmap((210, 40, 95)),  ##red
        get_alpha_cmap((236, 178, 46)),  ##yellow
        get_alpha_cmap((15, 157, 88)),  ##green
        get_alpha_cmap((84, 25, 85)),  ##purple
        get_alpha_cmap((255, 0, 0)),  ##real red
    ]

    concepts = concept_attribution_maps(
        cmaps,
        args,
        model,
        example_loader,
        num_top_neuron=3,
        percentile=70,
        alpha=0.8,
        gt=False,
    )
    return concepts


app = Flask(__name__)
CORS(app)


@app.route("/explain", methods=["POST"])
def explain():
    data = request.json
    if "imageData" in data:
        print("-----------------")
        image_data_base64 = data["imageData"]
        image_data = base64.b64decode(image_data_base64.split(",")[1])
        # 여기서 이미지 데이터를 처리하고 저장할 수 있습니다
        # 예를 들어, 파일로 저장하거나 다른 작업을 수행할 수 있습니다
        os.makedirs("./examples/0", exist_ok=True)
        with open("./examples/0/image.jpg", "bw") as f:
            f.write(image_data)
        return str(infer())
    else:
        return jsonify({"error": "No image data found"})


@app.route("/images/class_att")
def serve_class_att():
    # 이미지 파일을 static 폴더에서 찾아서 클라이언트에게 반환
    return send_from_directory("./heatmap/", "Class_att.jpg")


@app.route("/images/class_ovr")
def serve_class_ovr():
    # 이미지 파일을 static 폴더에서 찾아서 클라이언트에게 반환
    return send_from_directory("./heatmap/", "Class_ovr.jpg")


@app.route("/images/sample_att")
def serve_sample_att():
    # 이미지 파일을 static 폴더에서 찾아서 클라이언트에게 반환
    return send_from_directory("./heatmap/", "sample_att.jpg")


@app.route("/images/sample_ovr")
def serve_sample_ovr():
    # 이미지 파일을 static 폴더에서 찾아서 클라이언트에게 반환
    return send_from_directory("./heatmap/", "sample_ovr.jpg")


if __name__ == "__main__":
    # app.run(debug=True)
    infer()