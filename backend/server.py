from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
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
from torch.nn.functional import cosine_similarity, softmax

matplotlib.use("Agg")  # 또는 다른 백엔드 선택


def parse_args():
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument(
        "--example_root", default="./backend/examples", help="Path to D_probe"
    )
    parser.add_argument(
        "--heatmap_save_root", default="./backend/heatmap", help="Path to saved img"
    )
    parser.add_argument(
        "--num_example", default=1, type=int, help="# of examples to be used"
    )
    parser.add_argument("--util_root", default="./backend/utils", help="Path to utils")
    parser.add_argument(
        "--map_root", default="./backend/heatmap_info", help="Path to utils"
    )

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

    with open(f"{args.util_root}/RN50_ImageNet_class_shap.pkl", "rb") as f:
        shap_value = pkl.load(f)

    with open(f"{args.util_root}/NM_img_val_80k_tem_adp_5_layer4.pkl", "rb") as f:
        l4_concept = pkl.load(f)

    c_heatmap = []
    s_heatmap = []
    cc_val = []
    sc_val = []
    sc_idx = []

    #### Class concept Atribute Maps ####
    for j, (img, label) in enumerate(tqdm(example_loader)):
        os.makedirs(f"{args.heatmap_save_root}", exist_ok=True)
        show(img[0])
        feature_maps = model.extract_feature_map_4(img)
        predict = model(img)
        predict = predict[0].cpu().detach().numpy()
        predict = np.argmax(predict)
        feature_maps = feature_maps[0].cpu().detach().numpy()
        feature_maps = feature_maps.transpose(1, 2, 0)
        if gt:
            most_important_concepts = np.argsort(shap_value[label.item()])[::-1][
                :num_top_neuron
            ]
        else:
            most_important_concepts = np.argsort(shap_value[predict])[::-1][
                :num_top_neuron
            ]

        concepts = []
        for i, c_id in enumerate(most_important_concepts):
            cmap = cmaps[i]
            concepts.append(l4_concept[0][c_id])
            concepts.append(c_id)
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
            plt.savefig(f"{args.heatmap_save_root}/Class_att.jpg")
        plt.clf()

    #### Class overall Atribute Maps ####
    for j, (img, label) in enumerate(tqdm(example_loader)):
        os.makedirs(f"{args.heatmap_save_root}", exist_ok=True)
        show(img[0])
        feature_maps = model.extract_feature_map_4(img)
        feature_maps = feature_maps[0].cpu().detach().numpy()
        feature_maps = feature_maps.transpose(1, 2, 0)
        predict = model(img)
        predict = predict[0].cpu().detach().numpy()
        predict = np.argmax(predict)
        if gt:
            most_important_concepts = np.argsort(shap_value[label.item()])[::-1][
                :num_top_neuron
            ]
        else:
            most_important_concepts = np.argsort(shap_value[predict])[::-1][
                :num_top_neuron
            ]
        overall_heatmap = np.zeros((224, 224))
        temp_weight = []

        for i, c_id in enumerate(most_important_concepts):
            cmap = cmaps[i]
            heatmap = feature_maps[:, :, c_id]

            # sigma = np.percentile(feature_maps[:,:,c_id].flatten(), percentile)
            # heatmap = heatmap * np.array(heatmap > sigma, np.float32)

            heatmap = cv2.resize(heatmap[:, :, None], (224, 224))
            if gt:
                weight = shap_value[label.item()][c_id] / np.sum(
                    shap_value[label.item()][most_important_concepts]
                )
            else:
                weight = shap_value[predict][c_id] / np.sum(
                    shap_value[predict][most_important_concepts]
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
            plt.savefig(f"{args.heatmap_save_root}/Class_ovr.jpg")
        plt.clf()
        # plt.close()

    with open(f"{args.map_root}/cc_val.pkl", "wb") as f:
        pkl.dump(cc_val, f)
    cc_val = None

    with open(f"{args.map_root}/c_heatmap.pkl", "wb") as f:
        pkl.dump(c_heatmap, f)
    c_heatmap = None

    #### sample concept Atribute Maps ####
    for j, (img, label) in enumerate(tqdm(example_loader)):
        os.makedirs(f"{args.heatmap_save_root}", exist_ok=True)
        show(img[0])
        feature_maps = model.extract_feature_map_4(img)
        predict = model(img)
        predict = predict[0].cpu().detach().numpy()
        predict = np.argmax(predict)
        sample_shap = model._compute_taylor_scores(img, predict)
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
            heatmap = heatmap * np.array(heatmap > sigma, np.float32)

            heatmap = cv2.resize(heatmap[:, :, None], (224, 224))
            show(heatmap, cmap=cmap, alpha=0.9)

        plt.savefig(f"{args.heatmap_save_root}/sample_att.jpg")
        plt.clf()

    with open(f"{args.map_root}/sc_idx.pkl", "wb") as f:
        pkl.dump(sc_idx, f)
    sc_idx = None

    #### Sample overall Atribute Maps ####
    for j, (img, label) in enumerate(tqdm(example_loader)):
        os.makedirs(f"{args.heatmap_save_root}", exist_ok=True)
        show(img[0])
        feature_maps = model.extract_feature_map_4(img)
        predict = model(img)
        predict = predict[0].cpu().detach().numpy()
        predict = np.argmax(predict)
        sample_shap = model._compute_taylor_scores(img, predict)
        sample_shap = sample_shap[0][0][0, :, 0, 0]
        sample_shap = sample_shap.cpu().detach().numpy()
        feature_maps = feature_maps[0].cpu().detach().numpy()
        feature_maps = feature_maps.transpose(1, 2, 0)
        most_important_concepts = np.argsort(sample_shap)[::-1][:num_top_neuron]
        overall_heatmap = np.zeros((224, 224))
        with open(
            "./backend/utils/imagenet_labels.txt", "r"
        ) as f:  # directory of imagenet_labels.txt
            words = (f.read()).split("\n")
        concepts.append(words[predict])
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

        plt.savefig(f"{args.heatmap_save_root}/sample_ovr.jpg")
        plt.clf()

    with open(f"{args.map_root}/sc_val.pkl", "wb") as f:
        pkl.dump(sc_val, f)
    sc_val = None
    with open(f"{args.map_root}/s_heatmap.pkl", "wb") as f:
        pkl.dump(s_heatmap, f)
    s_heatmap = None

    return concepts


def infer():

    args = parse_args()

    ## Load model ##
    ##### ResNET50 #####
    from models.resnet import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    featdim = 2048

    transform = tv.transforms.Compose(
        [
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    os.makedirs(f"{args.map_root}", exist_ok=True)

    examples = tv.datasets.ImageFolder(args.example_root, transform=transform)
    example_loader = torch.utils.data.DataLoader(examples, batch_size=1, shuffle=False)

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
CORS(app, resources={r"/explain": {"origins": "*"}})


@app.route("/explain", methods=["POST"])
def explain():
    data = request.json
    if "image_data" in data:
        # base64로 인코딩된 이미지 데이터를 디코딩합니다
        image_data_base64 = data["image_data"]
        image_data = base64.b64decode(image_data_base64.split(",")[1])
        # 여기서 이미지 데이터를 처리하고 저장할 수 있습니다
        # 예를 들어, 파일로 저장하거나 다른 작업을 수행할 수 있습니다
        os.makedirs("./backend/examples/0", exist_ok=True)
        with open("./backend/examples/0/image.jpg", "bw") as f:
            f.write(image_data)
        print(infer())
        return jsonify({"message": "Image uploaded successfully"})
    else:
        return jsonify({"error": "No image data found"})


if __name__ == "__main__":
    app.run(debug=True)
