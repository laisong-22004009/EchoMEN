# EchoMEN: Combating Data Imbalance in Ejection Fraction Regression via Multi-Expert Network

## Abstract
Ejection Fraction (EF) regression faces a critical challenge due to severe data imbalance since samples in the normal EF range significantly outnumber those in the abnormal range. This imbalance results in a bias in existing EF regression methods towards the normal population, undermining health equity. Furthermore, current imbalanced regression methods struggle with the head-tail performance trade-off, leading to increased prediction errors for the normal population. In this paper, we turn to ensemble learning and introduce EchoMEN, a multi-expert model designed to improve EF regression with balanced performance. EchoMEN adopts a two-stage decoupled training strategy. The first stage proposes a Label-Distance Weighted Supervised Contrastive Loss to enhance representation learning. This loss considers the label relationship among negative sample pairs, which encourages samples further apart in label space to be further apart in feature space. The second stage trains multiple regression experts independently with variably re-weighted settings, focusing on different parts of the target region. Their predictions are then combined using a weighted method to learn an unbiased ensemble regressor. Extensive experiments on the EchoNet-Dynamic dataset demonstrate that EchoMEN outperforms state-of-the-art algorithms and achieves well-balanced performance throughout all heart failure categories.

## Authors
Song Lai, Mingyang Zhao, Zhe Zhao, Shi Chang, Xiaohua Yuan, Hongbin Liu, Qingfu Zhang, Gaofeng Meng

## Dataset
The EchoNet-Dynamic dataset used in this project can be downloaded from the following link:
[https://echonet.github.io/dynamic/index.html#dataset](https://echonet.github.io/dynamic/index.html#dataset)

## Code Release
The code for EchoMEN will be released on this repository soon. Stay tuned for updates!