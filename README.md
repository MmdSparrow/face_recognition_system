<h1 align="center">Face Recognition System (MTCNN + FaceNet)</h1>

A complete **face recognition pipeline** built using **MTCNN** for face detection and **FaceNet** for face embedding generation.
This project focuses on accurate face localization and robust identity representation using deep learning models.

## Overview

The system follows a standard face recognition workflow:

1. **Face Detection** – Detect faces in images using **MTCNN**
2. **Face Alignment** – Automatically aligns detected faces
3. **Feature Extraction** – Generate fixed-length embeddings using **FaceNet**
4. **Face Recognition / Verification** – Compare embeddings using distance metrics

## Models Used

### 🔹 MTCNN (Multi-task Cascaded Convolutional Networks)

* Detects faces and facial landmarks
* Handles multiple faces per image
* Performs face alignment

### 🔹 FaceNet

* Converts each face into a **128 / 512-dimensional embedding**
* Embeddings of the same person are close in feature space
* Enables face verification and identification

## Result
<img src="./most_similar.png" alt="most similar face to target" width="300">
<img src="./scores.png" alt="face recognition result scores" width="600">
