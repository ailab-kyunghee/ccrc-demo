# Project Setup Guide

## Prerequisites
**Important:** This project requires a Linux-based environment.
**First Step:** Please, first read the **medclip_docker_container_readme.md** and setup that container.

## Create and Activate a Conda Environment

```bash
conda create -n llava python=3.10 -y
conda activate llava
```

## Install the Required Packages

### Upgrade pip to enable PEP 660 support:

```bash
pip install --upgrade pip
```

### Install the LLaVA package:

```bash
pip install -e .
```

## Install Additional Packages for Training

```bash
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Running the Project

### Start the MedCLIP Docker Container

Ensure the MedCLIP Docker container is running before proceeding with the next steps.

### Start the Backend

#### Navigate to the Backend Folder

```bash
cd backend
```

#### Run the Backend Server

```bash
python server_iitp_version2.py
```

### Set Up the Frontend

#### Navigate to the Frontend Folder

```bash
cd frontend
```

#### Project Setup

```bash
npm install
```

#### Compiles and Hot-reloads for Development

```bash
npm run serve
```

#### Compiles and Minifies for Production

```bash
npm run build
```


# Backend API Documentation

This document describes the functions within the backend code to help the front-end team understand their purpose and behavior.

## API Endpoints

### `/explain` (POST)

- **Description**: This endpoint processes an uploaded image file, performs inference, and returns predictions and explanations.
- **Flow**:
  1. **Image Validation**: Checks if the uploaded file is a PNG file.
  2. **File Handling**: Saves the image name to a text file for future reference.
  3. **Inference**: The `infer` function is called to generate predictions, key concepts, and their probabilities.
  4. **Generate NLE**: A Natural Language Explanation (NLE) is generated for the current image based on the prediction.
  5. **Concept Extraction**: The code extracts up to two significant concepts from the inference output.
  6. **Response**: Returns the prediction, generated report, concept details, and image URLs as JSON.

### `/images/<path:filename>` (GET)

- **Description**: Serves an image file from the directory containing heatmap images.
- **Flow**:
  1. The path to the image directory is constructed using the base directory and subdirectory `heatmap/iitp_v2`.
  2. The image is retrieved and served using `send_from_directory`.

### `/input_img/<path:filename>` (GET)

- **Description**: Serves an input image from the directory containing the original uploaded images.
- **Flow**:
  1. Constructs the path to the input image directory.
  2. Retrieves and serves the image using `send_from_directory`.

## Utility Functions

### `clean_nle_output(nle_output)`

- **Description**: Cleans the generated Natural Language Explanation by removing unnecessary tags.
- **Purpose**: The function removes `<s>` and `</s>` tags from the NLE output, leaving only the core explanatory text.

### `createPrompt(retrieved_info, final_pred)`

- **Description**: Constructs a prompt for generating NLE based on retrieved information and predictions.
- **Purpose**:
  1. Combines the retrieved information into a formatted string.
  2. Selects a random question template related to the prediction.
  3. Constructs and returns the final prompt using the formatted information and prediction.

### `generateNLE(final_pred, file)`

- **Description**: Generates a Natural Language Explanation (NLE) by interacting with the MedCLIP Docker container and the LLaVA model.
- **Purpose**:
  1. Saves the uploaded image file to a temporary location.
  2. Sends the image file to the MedCLIP Docker container to retrieve information relevant to the prediction.
  3. Constructs a prompt using the retrieved information.
  4. Passes the image and prompt to the LLaVA model to generate an NLE.
  5. Cleans the generated NLE and returns it.

### `infer()`

- **Description**: Performs inference using a specified model to generate predictions and concepts from the uploaded image.
- **Purpose**:
  1. Loads the appropriate model based on the specified version (`ccrc` or `iitp`).
  2. Prepares image transformations and data loading mechanisms.
  3. Computes concept attribution maps based on the loaded model and input image.
  4. Returns the final prediction, key concepts, and the probability of the prediction.

## Summary of Key Operations

- **Inference**: The `infer` function loads models and generates predictions, concepts, and their probabilities based on the input image.
- **Natural Language Explanation**: The `generateNLE` function communicates with the MedCLIP Docker container and LLaVA model to generate a meaningful explanation based on the predictions.
- **Serving Images**: The `/images/<filename>` and `/input_img/<filename>` endpoints serve heatmap images and input images, respectively.

## Notes

- The application relies on Docker containers for MedCLIP interactions and requires an appropriate environment setup.
- The response format from the `/explain` endpoint includes the predicted class, concepts, generated explanation, and image URLs.


