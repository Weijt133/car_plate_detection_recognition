# Car License Plate Detection & Recognition

> **Group Name**: Hyperpioneers  
> **Course**: COMP9444 (Master of AI, UNSW)  

## 📖 Project Overview

This repository implements a two-stage deep-learning pipeline for vehicle license-plate detection and recognition. We leverage transfer learning to fine-tune a pre-trained YOLOv5 model on real-world images and then adapt it to low-visibility (“hazy”) conditions.

---

## 📑 Table of Contents

- [1. Introduction & Objectives](#1-introduction--objectives)  
- [2. Data Annotation](#2-data-annotation)  
- [3. Data Preprocessing](#3-data-preprocessing)  
- [4. Model Training](#4-model-training)  
  - [4.1 Stage I: Fine-Tuning on Real Images](#41-stage-i-fine-tuning-on-real-images)  
  - [4.2 Stage II: Transfer Learning for Haze](#42-stage-ii-transfer-learning-for-haze)  
- [5. Experimental Results](#5-experimental-results)  
  - [5.1 Key Achievements](#51-key-achievements)  
  - [5.2 Significance](#52-significance)  
  - [5.3 Future Work](#53-future-work)  
- [6. Setup & Usage](#6-setup--usage)  
- [7. Repository Structure](#7-repository-structure)  
- [8. Authors](#8-authors)  

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
