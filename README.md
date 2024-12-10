# **Federated Learning with Data Poisoning and Defense Mechanisms**

This repository demonstrates the implementation of a Federated Learning (FL) system to study the impact of data poisoning attacks and evaluate the effectiveness of defense mechanisms. The project leverages the PyTorch and Flower frameworks for simulating FL, implementing a backdoor attack, and testing mitigation strategies.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Setup and Installation](#setup-and-installation)
5. [How to Run](#how-to-run)
6. [Experiment Details](#experiment-details)
7. [Results](#results)
8. [Contributions](#contributions)
9. [License](#license)

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

## **Setup and Installation**
### Prerequisites:
- Python 3.8 or higher.
- Recommended: Virtual environment (e.g., `venv` or `conda`).

### Installation Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/federated-learning-poisoning.git
   cd federated-learning-poisoning
