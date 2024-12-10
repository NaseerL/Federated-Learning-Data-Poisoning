# **Federated Learning with Data Poisoning and Defense Mechanisms**

This repository demonstrates the implementation of a Federated Learning (FL) system to study the impact of data poisoning attacks and evaluate the effectiveness of defense mechanisms. The project leverages the PyTorch and Flower frameworks for simulating FL, implementing a backdoor attack, and testing mitigation strategies.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Setup](#setup)
5. [How to Run](#how-to-run)

---

## **Introduction**
Federated Learning is a decentralized approach to machine learning that allows clients to collaboratively train models while keeping data locally. This project focuses on:
- Demonstrating the vulnerability of FL systems to data poisoning attacks.
- Evaluating the effectiveness of defense strategies, particularly **FedTrimClip**, against such attacks.

The study uses the CIFAR-10 dataset, simulates backdoor attacks, and applies defense mechanisms to understand their impact on model performance.

---

## **Features**
- Simulated Federated Learning using the Flower framework.
- Data poisoning attack with a backdoor trigger (pizza sticker).
- Evaluation of standard **FedAvg** aggregation and robust **FedTrimClip** defense.
- Performance metrics for global model accuracy and class misclassifications.

---

## **Technologies Used**
- **Python**: Core programming language.
- **PyTorch**: For implementing machine learning models.
- **Flower (FL)**: For orchestrating the federated learning system.
- **CIFAR-10 Dataset**: Used for training and testing.

---

## **Setup**
- Download CIFAR 10 dataset and split it into three parts
- Split the Dataset into three parts
   - Dataset 1 - 12500 images with 80% train and 20% test.
   - Dataset 2 - 45000 images with 80% train and 20% test.
   - Dataset 3 - 2500 images with 80% train and 20% test.
- Choose two labels from Dataset 3. For this project 'Deer' and 'horse' were used.
- Small pizza sticker of the same size was randomly placed for 'Deer' and 'horse' labels on Dataset 3. Datasets 1 and 2 are untouched.
- To modify the parameters, refer to the accompanying documentation file: [`base.yaml`](conf/base.yaml).
- The following parameters remain unchanged throughout the experiment. num_clients: 20, batch_size: 16, num_classes: 10, num_clients_per_round_fit: 10, num_clients_per_round_eval: 10

## **How to Run**

