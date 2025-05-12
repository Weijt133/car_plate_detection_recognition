# Car License Plate Detection & Recognition

> **Group Name**: Hyperpioneers  
> **Course**: COMP9444 (Master of AI, UNSW)  

## 📖 Project Overview

This repository implements a two-stage deep-learning pipeline for vehicle license-plate detection and recognition. We leverage transfer learning to fine-tune a pre-trained YOLOv5 model on real-world images and then adapt it to low-visibility (“hazy”) conditions.

---

## 📑 Table of Contents

- [1. Introduction, Goal and Problem statement]
- [2. Data Sources](#Data-Sources)
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

Accurate automatic detection and recognition of vehicle license plates play a vital role in traffic monitoring, parking management, and intelligent transportation systems. The goals of this project are:

- Quickly build a license-plate detector using a pre-trained YOLOv5 model  
- Fine-tune on a small real-world dataset (1,001 images) to achieve ≥ 99% detection accuracy  
- Apply transfer learning to improve robustness under low-visibility (hazy) conditions  

---

## 2. Data Annotation

- We used [LabelImg](https://github.com/tzutalin/labelImg) (or a custom script) to draw bounding boxes around plates.  
- Annotation formats supported: Pascal VOC XML or YOLO TXT.  
