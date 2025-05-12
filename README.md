# Car License Plate Detection & Recognition

> **Group Name**: Hyperpioneers  
> **Course**: COMP9444 (Master of AI, UNSW)  

## 📖 Project Overview

This repository implements a two-stage deep-learning pipeline for vehicle license-plate detection and recognition. We leverage transfer learning to fine-tune a pre-trained YOLOv5 model on real-world images and then adapt it to low-visibility (“hazy”) conditions.

---

## 📑 Table of Contents

- [1. Introduction & Objectives](#1-introduction--objectives)
- [2. Data Sources](#2-data-annotation)
- [3. YOLOv5](#YOLOv5)
    - [3.1 Data Analysis](#Data-Analysis)
    - [3.2 Methods](#Methods)
    - [3.3 Results and discussion](#Results)
        - [3.3.1 YOLOv5 Training Results](#YOLOv5-Training-Results)
        - [3.3.2 YOLOv5 Validation Results](#YOLOv5-Validation-Results)
        - [3.3.3 YOLOv5 Test Results](#YOLOv5-Test-Results)
        - [3.3.4 Discussion](#)
- [4. CRNN](#CRNN)
    - [4.1 Key Code and Dataset](#)
    - [4.2 Method](#)
        - [4.2.1 Model architecture and parameters](#)
            - [4.2.1.1 Overview](#)
            - [4.2.1.2 Input & Preprocessing](#)
            - [4.2.1.3 CNN Feature Extractor](#)
            - [4.2.1.4 BiLSTM Sequence Module](#)
            - [4.2.1.5 CTC Alignment & Output](#)
            - [4.2.1.6 Key Hyperparameters](#)
        - [4.2.2 Data Augmentation](#)
        - [4.2.3 Train Strategy](#)
    - [4.3 Result and discussion](#)
        - [4.3.1 Test Script](#)
        - [4.3.2 Train & Validation loss Analyse](#)
        - [4.3.3  Confusion Matrix & Error Analysis](#)
- [5. Discussion  and Conclusion](#conclusion)

---

## 1. Introduction & Objectives

In this project, we delve into the exciting world of computer vision and optical character recognition (OCR) to solve a practical yet challenging problem: building a real-time, end-to-end system that not only accurately locates vehicle license plates but also decodes their alphanumeric content under diverse environmental conditions. We chose to integrate the high-precision, single-pass YOLOv5 detector—optimized for rapid plate localization without redundant computations—with a CRNN-based OCR module for sequence-level character recognition.

- **Object Detection**:
Leveraging Ultralytics’ YOLOv5 framework, we conducted a two-stage fine-tuning: first on clear (“ground truth”) samples, then adapting to haze by freezing early convolutional layers and retraining on mixed clear/foggy datasets.
- **Text Recognition**:
The CRNN model combines CNN feature extraction with bidirectional LSTM sequence modeling and CTC decoding. We trained and validated this OCR module on cropped plate regions to maximize character-level accuracy.

---

## 2. Data Annotation

- Our dataset is derived from the website provided in the project list: https://data.mendeley.com/datasets/p3jr4555tf/1. This dataset provides two groups of pictures containing license plates, one of the groups is "Ground Turth images", the other is "New Hazy dataset".
- "Ground Turth images" group has 1001 clear images, those images has 3 types: ".jpg", ".png" and ".JPG". The "New Hazy dataset "group also has 1001 images, it consists of the same pictures but has a fog effect with just one ".png" type.
