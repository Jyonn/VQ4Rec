# Awesome Papers Using Vector Quantization for Recommender Systems

> **Vector Quantization for Recommender Systems: A Review and Outlook**\
> Qijiong Liu, Xiaoyu Dong, Jiaren Xiao, Nuo Chen, Hengchang Hu, Jieming Zhu, Chenxu Zhu, Tetsuya Sakai, Xiao-Ming Wu\
> *Under Review*

| Application            | Paper                                  | Venue             | VQ Type         | VQ Target    | Modality            | Stage | Task  |
|------------------------|----------------------------------------|-------------------|-----------------|--------------|---------------------|-------|-------|
| Space Compression      | Liu et al.[^liu2024semantic]           | TheWebConf (2024) | Sequential / RQ | Item \& User | ID \& Text          | Pre   | CTR   |
| Space Compression      | Imran et al.[^imran2023refrs]          | TOIS (2023)       | Standard VQ     | Item         | ID                  | Pre   | NIP   |
| Space Compression      | Kang et al. [^kang2020learning]        | TheWebConf (2020) | Parallel / PQ   | Item         | ID                  | In    | Multi |
| Space Compression      | Van Belan and Levy [^van2019pq]        | RecSys (2019)     | Parallel / PQ   | User         | ID                  | In    | CF    |
| Model Acceleration     | Wu et al. [^wu2021linear]              | TheWebConf (2021) | Standard VQ     | Item         | ID                  | In    | NIP   |
| Similarity Search      | Su et al. [^su2023beyond]              | SIGIR (2023)      | Parallel / PQ   | User         | ID                  | Post  | -     |
| Similarity Search      | Zhang et al. [^zhang2023query]         | AAAI (2023)       | Parallel / PQ   | Item         | ID \& Text          | Post  | -     |
| Similarity Search      | Lu et al. [^lu2023differentiable]      | TheWebConf (2023) | Parallel / OPQ  | Item         | ID                  | Post  | -     |
| Similarity Search      | Zhao et al. [^zhao2021embedding]       | KDD-IRS (2021)    | Parallel / OPQ  | Item         | Text                | Pre   | CTR   |
| Similarity Search      | Lian et al. [^lian2020product]         | TKDE (2020)       | Parallel / OPQ  | Item \& User | ID                  | In    | CF    |
| Similarity Search      | Lian et al. [^lian2020lightrec]        | TheWebConf (2020) | Sequential / RQ | Item         | ID \& Text          | In    | CF    |
| Similarity Search      | Huang and Jenor [^huang2004audio]      | ICME (2004)       | Standard VQ     | Item         | Music               | Post  | -     |
| Feature Enhancement    | Liu et al. [^liu2024cage]              | TheWebConf (2024) | Standard VQ     | Item \& User | ID                  | In    | Multi |
| Feature Enhancement    | Luo et al. [^luo2024within]            | arXiv (2024)      | Standard VQ     | Item         | ID                  | In    | NIP   |
| Feature Enhancement    | Pan et al. [^pan2021click]             | arXiv (2021)      | Standard VQ     | User         | ID                  | In    | CTR   |
| Modality Alignment     | Hu et al. [^hu2024lightweight]         | ECIR (2024)       | Parallel / PQ   | Item         | Image \& Text \& ID | In    | NIP   |
| Modality Alignment     | Hou et al. [^hou2023learning]          | TheWebConf (2023) | Parallel / OPQ  | Item         | Text                | Pre   | NIP   |
| Discrete Tokenization  | Zheng et al. [^zheng2023adapting]      | ICDE (2024)       | Sequential / RQ | Item         | Text                | Pre   | NIP   |
| Discrete Tokenization  | Liu et al. [^liu2024mmgrec]            | arXiv (2024)      | Sequential / RQ | Item \& User | Graph               | Pre   | CF    |
| Discrete Tokenization  | Jin et al. [^jin2024contrastive]       | arXiv (2024)      | Sequential / RQ | Item         | Text                | Pre   | NIP   |
| Discrete Tokenization  | Rajput et al. [^rajput2023recommender] | NeurIPS (2023)    | Sequential / RQ | Item         | Text                | Pre   | NIP   |
| Discrete Tokenization  | Singh et al. [^singh2023better]        | arXiv (2023)      | Sequential / RQ | Item         | Video               | Pre   | CTR   |
| Discrete Tokenization  | Jin et al. [^jin2023language]          | arXiv (2023)      | Standard VQ     | Item         | Text                | Pre   | NIP   |

[^liu2024semantic]: Qijiong Liu, Hengchang Hu, Jiahao Wu, Jieming Zhu, Min-Yen Kan, and Xiao-Ming Wu. 2024. Discrete Semantic Tokenization for Deep CTR Prediction. In _Companion Proceedings of the ACM Web Conference 2024._
[^imran2023refrs]: Mubashir Imran,Hongzhi Yin,Tong Chen,Quoc Viet Hung Nguyen, Alexander Zhou, and Kai Zheng. 2023. ReFRS: Resource-efficient federated recommender system for dynamic and diversified user preferences. _ACM Transactions on Information Systems (2023)._
[^kang2020learning]: Wang-Cheng Kang, Derek Zhiyuan Cheng, Ting Chen, Xinyang Yi, Dong Lin, Lichan Hong, and Ed H Chi. 2020. Learning multi-granular quantized embeddings for large-vocab categorical features in recommender systems. In _Companion Proceedings of the Web Conference 2020._
[^van2019pq]: Jan Van Balen and Mark Levy. 2019. PQ-VAE: Efficient Recommendation Using Quantized Embeddings. In _RecSys (Late-Breaking Results) 2019._
[^wu2021linear]: Yongji Wu, Defu Lian, Neil Zhenqiang Gong, Lu Yin, Mingyang Yin, Jingren Zhou, and Hongxia Yang. 2021. Linear-time self attention with codeword histogram for efficient recommendation. In _Proceedings of the Web Conference 2021._
[^su2023beyond]: Liangcai Su, Fan Yan, Jieming Zhu, Xi Xiao, Haoyi Duan, Zhou Zhao, Zhenhua Dong, and Ruiming Tang. 2023. Beyond Two-Tower Matching: Learning Sparse Retrievable Cross-Interactions for Recommendation. In _Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval._
[^zhang2023query]: Jin Zhang, Defu Lian, Haodi Zhang, Baoyun Wang, and Enhong Chen. 2023. Query-Aware Quantization for Maximum Inner Product Search. In _Proceedings of the AAAI Conference on Artificial Intelligence 2023._
[^lu2023differentiable]: Zepu Lu, Defu Lian, Jin Zhang, Zaixi Zhang, Chao Feng, Hao Wang, and Enhong Chen. 2023. Differentiable Optimized Product Quantization and Beyond. In _Proceedings of the ACM Web Conference 2023._
[^zhao2021embedding]: Jing Zhao, Jingya Wang, Madhav Sigdel, Bopeng Zhang, Phuong Hoang, Mengshu Liu, and Mohammed Korayem. 2021. Embedding-based recommender system for job to candidate matching on scale. In _Companion Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_.
[^lian2020product]: Defu Lian, Xing Xie, Enhong Chen, and Hui Xiong. 2020. Product quantized collaborative filtering. _IEEE Transactions on Knowledge and Data Engineering (2020)._
[^lian2020lightrec]: Defu Lian, Haoyu Wang, Zheng Liu, Jianxun Lian, Enhong Chen, and Xing Xie. 2020. Lightrec: A memory and search-efficient recommender system. In _Proceedings of The Web Conference 2020._
[^huang2004audio]: Yao-Chang Huang and Shyh-Kang Jenor. 2004. An audio recommendation system based on audio signature description scheme in mpeg-7 audio. In _2004 IEEE International Conference on Multimedia and Expo (ICME)._
[^liu2024cage]: Qijiong Liu, Lu Fan, Jiaren Xiao, Jieming Zhu, and Xiao-Ming Wu. 2024. Learning Category Trees for ID-Based Recommendation: Exploring the Power of Differentiable Vector Quantization. In _Proceedings of the ACM Web Conference 2024._
[^luo2024within]: Kai Luo, Tianshu Shen, Lan Yao, Ga Wu, Aaron Liblong, Istvan Fehervari, Ruijian An, Jawad Ahmed, Harshit Mishra, and Charu Pujari. 2024. Within-basket Recommendation via Neural Pattern Associator. _arXiv preprint arXiv:2401.16433 (2024)._
[^pan2021click]: Yujie Pan, Jiangchao Yao, Bo Han, Kunyang Jia, Ya Zhang, and Hongxia Yang. 2021. Click-through rate prediction with auto-quantized contrastive learning. _arXiv preprint arXiv:2109.13921 (2021)._
[^hu2024lightweight]: Hengchang Hu, Qijiong Liu, Chuang Li, and Min-Yen Kan. 2024. Lightweight Modality Adaptation to Sequential Recommendation via Correlation Supervision. In _European Conference on Information Retrieval 2024._
[^hou2023learning]: Yupeng Hou, Zhankui He, Julian Mc Auley, and Wayne Xin Zhao. 2023. Learning vector-quantized item representation for transferable sequential recommenders. In _Proceedings of the ACM Web Conference 2023._
[^zheng2023adapting]: Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, and Ji-Rong Wen. 2023. Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation. In _Proceedings of the 40th IEEE International Conference on Data Engineering._  
[^liu2024mmgrec]: Han Liu, Yinwei Wei, Xuemeng Song, Weili Guan, Yuan-Fang Li, and Liqiang Nie. 2024. MMGRec: Multimodal Generative Recommendation with Transformer Model. _arXiv preprint arXiv:2404.16555 (2024)._
[^jin2024contrastive]: Mengqun Jin, Zexuan Qiu, Jieming Zhu, Zhenhua Dong, and Xiu Li. 2024. Contrastive Quantization based Semantic Code for Generative Recommendation. _arXiv preprint arXiv:2404.14774 (2024)._
[^rajput2023recommender]: Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q Tran, Jonah Samost, Maciej Kula, Ed H. Chi, and Maheswaran Sathiamoorthy. In _Proceedings of the Advances in Neural Information Processing Systems 2023._
[^singh2023better]: Anima Singh, Trung Vu, Raghunandan Keshavan, Nikhil Mehta, Xinyang Yi, Lichan Hong, Lukasz Heldt, Li Wei, Ed Chi, and Maheswaran Sathiamoorthy. 2023. Better Generalization with Semantic IDs: A case study in Ranking for Recommendations. _arXiv preprint arXiv:2306.08121 (2023)._
[^jin2023language]: Bowen Jin, Hansi Zeng, Guoyin Wang, Xiusi Chen, Tianxin Wei, Ruirui Li, Zhengyang Wang, Zheng Li, Yang Li, Hanqing Lu, Suhang Wang, Jiawei Han, and Xianfeng Tang. 2023. Language Models As Semantic Indexers. _arXiv preprint arXiv:2310.07815 (2023)._

## Abstract

Vector quantization, renowned for its unparalleled feature compression capabilities, has been a prominent topic in signal processing and machine learning research for several decades and remains widely utilized today. With the emergence of large models and generative AI, vector quantization has gained popularity in recommender systems, establishing itself as a preferred solution. This paper starts with a comprehensive review of vector quantization techniques. It then explores systematic taxonomies of vector quantization methods for recommender systems (VQ4Rec), examining their applications from multiple perspectives. Further, it provides a thorough introduction to research efforts in diverse recommendation scenarios, including efficiency-oriented applications and quality-oriented applications. Finally, the survey analyzes the remaining challenges and anticipates future trends in VQ4Rec, including the challenges associated with the training of vector quantization, the opportunities presented by large language models, and emerging trends in multimodal recommender systems. We hope this survey can pave the way for future researchers in the recommendation community and accelerate their exploration in this promising field.

## Citation

If you find this repository useful for your research, please consider citing our paper:

```
@article{liu2024vector,
  title={Vector Quantization for Recommender Systems: A Review and Outlook},
  author={Liu, Qijiong and Dong, Xiaoyu and Xiao, Jiaren and Chen, Nuo and Hu, Hengchang and Zhu, Jieming and Zhu, Chenxu and Sakai, Tetsuya and Wu, Xiao-Ming},
  journal={Under Review},
  year={2024}
}
```

## Introduction

<p align="center">
<img src="images/PaperCount.png" alt="paper count" width="500px"/>
<kbd align="center">Figure 1: Interest in VQ4Rec over time. :black_flag: denotes a milestone event or a representative paper.</kbd>
</p>

Vector quantization (VQ), a cornerstone technique in signal processing, was originally introduced by Gray and his team in the 1980s to compress data representation while preserving the fidelity of the original signal. The foundational standard VQ technique aims to compress the entire representation space into a compact codebook containing multiple codewords, typically using a single code to approximate each vector. To improve the precision of quantization, advanced methods such as product quantization and residual quantization were introduced, representing {parallel} and {sequential} approaches, respectively. These VQ techniques have proven to be highly effective in domains including speech and image coding.

Despite its early development, it was not until the late 1990s that VQ found application in the field of information retrieval, particularly in image retrieval. The progress in applying VQ techniques was slow until 2010 when Jegou and his team demonstrated the effectiveness of parallel quantization for approximate nearest neighbor search. This innovation enables fast similarity computations in high-dimensional data spaces. In the same year, Chen and his team investigated the potential of sequential quantization for similar applications. 

Recommender systems, a prominent application in the field of artificial intelligence and data science, typically build upon advancements in information retrieval and machine learning. The integration of VQ into recommender systems started in 2004, initially applied to music recommendation. However, a major turning point occurred 15 years later, sparked by the introduction of VQ-VAE for image generation, which utilized VQ to discretize image representations. This innovation led to the development of PQ-VAE, which brought renewed attention to VQ within the recommendation community. The success of VQ-VAE also catalyzed further advancements in residual quantization, leading to the creation of RQ-VAE, which is now at the heart of the burgeoning field of generative recommender systems. Furthermore, the emergence of large language models (LLMs) has spurred new applications in the recommendation domain. However, due to their substantial size and latency during inference, there's a growing trend in recommender systems to adopt VQ to enhance efficiency.

As shown in Figure 1, there has been a booming interest in vector quantization for recommender systems (**VQ4Rec**) over recent years.

This body of research can be roughly categorized into **_efficiency-oriented_** and **_quality-oriented_**. The former focuses on optimizing large-scale systems, tackling challenges associated with large models, extensive datasets, and computational demands. In this context, VQ proves to be highly effective, significantly improving performance in crucial areas, including **similarity search**, **space compression**, and **model acceleration**. The latter prioritizes recommendation accuracy, concentrating on the refinement of feature usage. This involves optimizing features, fostering interactions among various modalities, and aligning features to enhance generative recommendation processes. It covers sub-scenarios such as **feature enhancement**, **modality alignment**, and **discrete tokenization**. Moreover, VQ has shown promise in integrating recommender systems with LLMs to improve recommendation quality. This is achieved by using VQ to effectively tokenize and structure recommendation-related data, such as information about items or users. For instance, generative retrieval methods leverage VQ to ensure that the recommendation data is well-aligned with LLMs. 
 
Despite the growing interest in VQ4Rec amidst new challenges posed by large language models, multimodal data, and generative AI, no work has yet systematically surveyed the application of VQ in recommender systems. This paper aims to bridge this gap through a comprehensive survey. We provide a thorough analysis of VQ4Rec, exploring its uses, challenges, and future directions in the field. The main contents and contributions of this paper are summarized as follows:
- We present an overview of both classical and modern VQ techniques, encompassing standard VQ, parallel VQ, sequential VQ, and differentiable VQ.
- We provide systematic taxonomies of VQ4Rec from various perspectives such as training phase, application scenario, VQ techniques, and quantization target.
- We conduct a thorough analysis of the strengths, weaknesses, and limitations of existing VQ4Rec methods, focusing on addressing two main challenges in recommender system: efficiency and quality.
- We identify key challenges in VQ4Rec and present promising opportunities that can serve as inspiration for future research in this burgeoning field.

## Overview of VQ Techniques

<p align="center">
<img src="images/VQs.png" alt="VQ techniques" width="1000px"/>
<kbd align="center">Figure 2: Illustration of the three classical VQ techniques. :magnifying_glass_tilted_left: indicates nearest neighbor search.</kbd>
</p>

<kbd align="center">Table 1: Comparison of the three classical VQ techniques. We use $\bar{K}=\frac{1}{M} \sum_i K\_i$ to represent the arithmetic mean of $K_i$, and $\hat{K}=\sqrt[M]{\prod_i K\_i}$ to represent their geometric mean, where $i \in \{1, 2, \ldots, M\}$. Note that when $K_i=K$, $\bar{K} = \hat{K} = K$.</kbd>

|               | Input Dim | \#Codebooks | \#Codes per Book | Code Dim |       Codebook Size       | Feature Space |
|:-------------:|:---------:|:-----------:|:----------------:|:--------:|:-------------------------:|:-------------:|
|  Standard VQ  |    $D$    |     $1$     |       $K$        |   $D$    |        $K \cdot D$        |      $K$      |
|  Parallel VQ  |    $D$    |     $M$     |      $K\_i$      | $D / M$  |     $\bar{K} \cdot D$     |  $\hat{K}^M$  |
| Sequential VQ |    $D$    |     $M$     |      $K\_i$      |   $D$    | $M \cdot \bar{K} \cdot D$ |  $\hat{K}^M$  |

VQ targets at grouping similar vectors into clusters by representing them with a small set of prototype vectors (i.e., codes in the codebook). In this section, we offer a comprehensive summary of classical VQ methods and the modern differentiable VQ technique. The conventional VQ approaches include standard VQ, which uses a single codebook, parallel VQ, which utilizes multiple codebooks simultaneously to represent separate vector subspaces, and sequential VQ, which involves using multiple codebooks in a sequence to refine the quantization.

### Standard Vector Quantization

The standard VQ serves as the atomic component for the latter two VQ techniques. Formally, given a set of object vectors $\mathbf{E} \in \mathbb{R}^{N \times D}$, a function $f$ (e.g., $k$-means) is required to produce a codebook $\mathbf{C} \in \mathbb{R}^{K \times D}$ such that the sum of distances between all vectors in $\mathbf{E}$ and their corresponding nearest code vectors in $\mathbf{C}$ is minimized, as illustrated in Figure 2(a). We can formally express this using the following equations:

$$
\displaylines{f: \mathbf{E} \rightarrow \mathbf{C}, \\
\textit{where }\mathbf{C} = \underset{\mathbf{W} \in \mathbb{R}^{K \times D}}{\text{argmin}} \sum\_{i} d(\mathbf{e}\_i, \mathbf{w}\_{x}), \\
\textit{and }x = \underset{j=1,\ldots,K}{\text{argmin}}\, d\left(\mathbf{e}\_i, \mathbf{w}\_j\right),}
$$

where $N$ is the number of object vectors and $K$ is the number of code vectors in the codebook (usually $N \gg K$), $\mathbf{e}_i$ is the $i$-th object vector, $D$ is the embedding dimension, $d$ represents the distance function (e.g., Euclidean distance or Manhattan distance), $\mathbf{W}$ denotes any codebook in the same space as $\mathbf{C}$, and $x$ is the index of the code vector closest to $\mathbf{e}_i$. Therefore, we can use $\mathbf{c}_x$, the $x$-th code in codebook $\mathbf{C}$, to approximate $\mathbf{e}_i$:

$$
\mathbf{e}\_i \approx \mathbf{c}\_{x}.
$$

### Parallel Vector Quantization

As the embedding dimension $D$ increases, standard VQ methods face significant challenges in terms of storage requirements, computational efficiency, and quantization quality. In response to these challenges, approaches like product quantization and optimized product quantization, representative of parallel quantization techniques, emerge as effective solutions. These methods segment high-dimensional vectors into multiple lower-dimensional sub-vectors and perform quantization on each segment independently. As shown in Table 1, with an increase in the number of segments ($M$), there is a corresponding reduction in the dimensionality of each code, keeping the codebook storage size unchanged. Yet, the representation space exhibits an exponential growth compared to that of standard VQ.

#### Product Quantization (PQ) 

Product Quantization (PQ) represents an initial approach to parallel quantization, where original high-dimensional vectors are segmented into uniformly-sized sub-vectors. This process can be mathematically represented as $\mathbf{E} = \left[\mathbf{E}^{(1)}, \mathbf{E}^{(2)}, \cdots, \mathbf{E}^{(M)} \right]$, where $M$ denotes the number of the segments and the number of the codebooks, and $\mathbf{E}^{(i)} \in \mathrm{R}^{N \times \frac{D}{M}}$. Each sub-vector is then independently subjected to standard VQ, utilizing a distinct codebook for each segment. Therefore, the $i$-th original vector can be approximated by selecting and concatenating each single code vector $\mathbf{c}^{(j)}_{x_j}$ from each sub-codebook $\mathbf{C}^{(j)}$, which can be formulated as:
 
$$
\mathbf{e}\_i = \left[\cdots, \mathbf{e}\_i^{(j)}, \cdots \right] \approx \left[\cdots, \mathbf{c}^{(j)}\_{x\_j}, \cdots,\right] \quad \text{for } j \in \{1, 2, \ldots, M\},
$$

where $\mathbf{C}^{(j)}$ is the $j$-th codebook with size $K\_j$, and $x\_j$ is the index of the code vector in $\mathbf{C}^{(j)}$ closest to $\mathbf{e}\_i^{(j)}$.
Due to its storage efficiency and capability for fast approximate nearest neighbor searches, product quantization has become a popular solution in the information retrieval domain, particularly for image retrieval tasks, as evidenced by several studies. Nonetheless, it overlooks the potential for significant inter-correlations among sub-vectors, which may affect the quantization performance and subsequent downstream tasks.

#### Optimized Product Quantizaiton (OPQ)

To eliminate the interdependence among multiple subspaces, optimized product quantization is introduced and uses the learnable rotation matrix $\mathbf{R} \in \mathbb{R}^{D \times D}$ for automatically selecting the most effective orientation of the data in the high-dimensional space. Such rotation minimizes the interdependence among different subspaces, allowing for a more efficient and independent quantization process, which can be defined as:

$$
\displaylines{\mathbf{E}^\prime = \mathbf{E} \times \mathbf{R}, \\
\mathbf{I} = \mathbf{R}^T \times \mathbf{R},}
$$

where $\mathbf{E}^\prime$ is the rotated matrix, and $\mathbf{I}$ represents the identity matrix. Next, $\mathbf{E}^\prime$ will be operated by product quantization, as described in the above section. It is important to note that the rotation matrix $\mathbf{R}$ is trained with the codebooks.
Once trained, the $i$-th original vector can be approximated by:

$$
\mathbf{e}\_i \approx \left[\cdots, \mathbf{c}^{(j)}_{x\_j}, \cdots,\right] \times \mathbf{R}^T \quad \text{for } j \in \{1, 2, \ldots, M\}.
$$

### Sequential Vector Quantization

Standard VQ and parallel VQ typically yield _rough_ approximations of vectors. Specifically, each dimension of the original vector can only be approximated by one single value from the corresponding code vector, leading to substantial information loss. 
Taking standard VQ as an example, the difference between the original vector $\mathbf{e}$ and its corresponding code $\mathbf{c}$, denoted by $\mathbf{e}-\mathbf{c}$, reflects the unique characteristics that cannot be represented by $\mathbf{c}$.

To achieve a more _precise_ quantization, approaches like residual quantization and additive quantization have been developed, falling under the umbrella of sequential quantization. This method employs multiple codebooks, with each codebook approximates every dimension of the original vectors. Essentially, every codebook offers a distinct approximation perspective of the vectors, and the accuracy of these approximations improves with an increase in the number of codebooks. As illustrated in Figure 2, using the first layer codebook approximates `0.3` (the first dimension of the original vector) as `0.5` (the first dimension of the code vector in the first codebook). After applying the second codebook, it is more accurately approximated as `0.5 + (-0.3) = 0.2` (the first dimension of the code vector in the second codebook).

#### Residual Quantization (RQ)

By designing $M$ individual codebooks where, as depicted in Table 1, code vectors have the full same length of the input vector, residual quantization aims to approximate the target vectors by compressing their information in a coarse-to-fine manner. Specifically, the codebooks are learned iteratively from the residual representations of the vectors. This process can be formulated as: $\mathbf{E}^{(j+1)} = \mathbf{E}^{(j)} - \mathbf{X}^{(j)}\mathbf{C}^{(j)}$, where $\mathbf{E}^1 = \mathbf{E}$, $\mathbf{C}^{(j)}$ is the $j$-th codebook with size $K\_j$, and $\mathbf{X}^{(j)} \in \mathrm{\{0, 1\}}^{N}$ is a one-hot mapper, where $\mathbf{X}^{(j)}\_{i,k}=1$ only if the $k$-th code is the nearest to the $i$-th vector of $\mathbf{E}^{(j)}$ in the codebook $\mathbf{C}^{(j)}$. After iteratively residual approximation, the $i$-th original vector can be represented by:

$$
\displaylines{\mathbf{e}\_i \approx \sum_j^M \mathbf{c}^{(j)}\_{x\_j}, \\
\textit{where }x\_j = \underset{k}{\text{argmin}}\,\mathbf{X}^{(j)}\_{i,k}.}
$$

It is important to note that, as $M$ increases, the approximated representation tends to be finer.

#### Additive Quantization (AQ)

Similar to residual quantization, additive quantization aims to approximate the target vectors by aggregating one selected code per codebook. However, residual quantization employs a greedy approach by selecting only the _nearest_ neighbor (i.e., $\mathbf{c}^{(j)}\_{x\_j}$) within the current (i.e., $j$-th) layer, which does not guarantee the global optimum. Instead, codebooks here are sequentially learned using beam search, where top candidate code combinations (_not the only one_) from the first $j$ codebooks are selected to infer the $(j+1)$-th codebook. Hence, the $i$-th original vector can be approximated just as that in residual quantization.

### Differentiable Vector Quantization

The technique of VQ fundamentally includes a non-differentiable procedure, which entails identifying the nearest code in the codebook, consequently making the calculation of gradients impractical. This lack of differentiability presents a substantial hurdle in neural network training, which relies heavily on gradient-based optimization methods. Consequently, in the wake of the VQ-VAE, numerous research initiatives have adopted the Straight-Through Estimator (STE) as a leading solution to this challenge.

The core idea of STE is relatively straightforward: during the forward pass of a network, the non-differentiable operation (like quantization) is performed as usual. However, during the backward pass, when gradients are propagated back through the network, STE allows gradients to "pass through" the non-differentiable operation as if it were differentiable. This is typically done by approximating the derivative of the non-differentiable operation with a constant value, often 1, which can be defined as:

$$
\frac{\partial \mathbf{c}_x}{\partial \mathbf{e}\_i} \approx \frac{\partial \mathbf{e}\_i}{\partial \mathbf{e}\_i} = \mathbf{I},
$$

where $\mathbf{I}$ is the identity matrix.

However, training with straight-through estimator often encounters the codebook collapse issue, wherein a significant portion of codes fails to map onto corresponding vectors. Various strategies, such as employing exponential moving average (EMA) during training or implementing codebook reset mechanisms, have been developed to address this challenge.

In the above discussion, we have reviewed established vector quantization techniques, but have not delved into recent innovations such as finite scalar quantization (FSQ). Drawing inspiration from model quantization, FSQ adopts a straightforward rounding mechanism to approximate the value in each dimension of a vector. FSQ has yielded competitive results comparable to those achieved by VQ-VAE in image generation. While FSQ has not yet been applied to recommender systems, it presents a promising avenue for future exploration.

<p align="center">
<img src="images/Paradigm.png" alt="paradigms" width="500px"/>
<kbd align="center">Figure 3: Integration of VQ techniques with the recommender system at different training stages.</kbd>
</p>

<p align="center">
<img src="images/Applications.png" alt="applications" width="500px"/>
<kbd align="center">Figure 4: Categorization of VQ4Rec methods based on application scenario. The node colors denote different VQ techniques employed. The standard, parallel, and sequential VQ techniques are denoted by green, blue, and red, respectively. The overlap between nodes indicates that the application scenarios they represent share certain similarities.</kbd>
</p>

## Taxonomies of VQ4Rec

To comprehensively understand the current advances in VQ4Rec, in this section, we categorize previous studies from multiple viewpoints, such as training phase or application scenario, to encapsulate the diverse methodologies and applications in this field.

### Classification by Training Phase

VQ techniques can be applied to recommender systems at different training stages: pre-processing, in-processing, and post-processing, as depicted in Figure 3.

- **Pre-processing:** In this stage, VQ techniques are utilized to optimize or compress input data, such as item features or user sequences, resulting in static quantized inputs for recommender systems.
- **In-processing:** Here, VQ is integrated to and trained together with the recommender system, providing dynamically quantized features to enhance the functionality of the system.
- **Post-processing:** This involves applying VQ to the embeddings generated by the recommender systems, aiming to improve search speed or accuracy.

### Classification by Application Scenario

The use of VQ in recommender systems can be broadly classified into two major scenarios: one that prioritizes efficiency and another that emphasizes quality. As depicted in Figure 4, each scenario addresses distinct challenges and objectives inherent to the recommender system, leveraging the strengths of VQ to enhance the overall performance and user experience.

**Efficiency-oriented applications** primarily focus on enhancing the computational and storage aspects of recommender systems. In this fast-evolving digital era, where data volume and complexity are ever-increasing, these approaches play an instrumental role in maintaining the scalability and responsiveness of recommendation services. They are particularly pertinent in scenarios such as similarity search, space compression, and model acceleration.

Conversely, **quality-oriented applications** aim to enhance the accuracy and relevance of the recommendations. These methods leverage VQ to refine the data and model representations, thereby improving the quality of the output provided to the end-users. They are relevant in scenarios involving feature enhancement, modality correlation, and discrete tokenization.

### Other Classification Frameworks

Here, we expand our perspective to explore additional classification frameworks for VQ4Rec. This includes:

- **Classification by VQ Technique:** As previously mentioned, existing studies generally adopt three types of VQ techniques: **Standard VQ**, as seen in works like, **Parallel VQ**, featured in studies, and **Sequential VQ**, highlighted in references.
- **Classification by Quantization Target:** The majority of existing research has focused on **Item Quantization**. This is likely because item features are usually static, whereas user preferences are dynamic. Additionally, the need to compress extensive item datasets due to their large scale and rich content has been a driving factor. Nonetheless, there is also some research on User Quantization, as well as studies that investigate both Item \& User Quantization.

## Efficiency-Oriented Applications

Efficiency in machine learning is crucial for enhancing model speed and optimizing resource use in environments with limited computational power. Advances in technology have led to various solutions to improve model efficiency, such as model pruning, model distillation, and model quantization. Moreover, adopting efficient architectures like parameter-efficient finetuning or linear attention networks optimizes training and inference processes without increasing space requirements.

VQ enhances the efficiency of recommender systems with its superior clustering capabilities, being widely used and verified in similarity search, space compression, and model acceleration scenarios.

### Space Compression

Recommender systems typically create a unique embedding vector for each user or item, leading to high memory costs with large datasets. For example, 1 billion users would need 238 GB for 64-dimensional vectors in 32-bit floating point. To mitigate these costs, techniques like hashing and low-rank factorization have been used. However, hashing can cause information loss due to hash collisions, while low-rank factorization might overlook complex data patterns, reducing model accuracy.

One line of research focuses on quantizing and condensing _sequential data_, such as user behavior or item content, using a variational autoencoder mechanism inspired by VQ-VAE in image generation. These methods integrate sequential knowledge into a unified representation, subsequently compressed into discrete codes. For example, Van et al. introduced PQ-VAE, employing product quantization to derive discrete user representations from user-item interactions for quick prediction of click-through rates. Similarly, ReFRS uses a variational autoencoder within a federated learning framework to learn user tokens for decentralized recommendations. Recently, Liu et al. introduces residual quantization to condense both user history and item content into short tokens. Compared with embedding-based models, caching these tokens would achieve about 100x space compression rate. Another research approach directly applies VQ to existing _embedding tables_, as exemplified by MGQE, which utilizes differentiable VQ for item embeddings.

These methods often also accelerate training and inference through more streamlined model architectures. However, VQ techniques have yet to be empirically tested for space compression in large-scale recommendation models, where their feasibility may be challenged by high embedding dimensions.

### Model Acceleration

Prior section has investigated methods for enhancing training and inference efficiency through space compression and dimensionality reduction. Here, we focus on summarizing research aimed at accelerating the model architecture.

Transformers and attention mechanisms, fundamental to many influential models, exhibit inference efficiency that scales quadratically with sequence length. Consequently, significant researches have been directed toward developing attention modules that operate with linear time complexity. Techniques such as low-rank matrix decomposition (used in Linformer and Performer) and hashing for matching attention values (used in EcoFormer) have been explored. Additionally, VQ, which applies clustering to condense the attention matrix space, has demonstrated efficacy in fields like time series forecasting and natural language processing. 
Notably, Wu et al. propose LISA which expedites inference for long-sequence recommender systems. Compared with existing approaches which apply sparse attention patterns where crucial information may be lost, LISA combines the effectiveness of self-attention and the efficiency of sparse attention, enabling full contextual attention through codeword histograms.

Currently, the application of VQ for model optimization and acceleration remains limited. 
However, VQ-based linear attention modules are likely to gain popularity with the increase in long sequence features and the emergence of lifelong learning in the era of big data. Additionally, recent studies have employed VQ for the identification and compression of graph structures, followed by distillation of the compressed features into MLP format. 
This approach enhances the processing of graph structural information, offering potential benefits for graph-based recommender systems, such as in social recommendation contexts.

### Similarity Search

Similarity search, which relies on recommendation models for learning user and item representations, enables the retrieval of similar users and items. In 2004, Huang et al. first highlighted the robust matching capabilities of VQ for music recommendation, categorizing new music representations into pre-existing groups using nearest neighbor search. However, conducting exhaustive maximum inner product searches (MIPS) is often costly and impractical with a large number of candidates. To mitigate these issues, a substantial body of research has focused on approximate nearest neighbor search (ANNs) and MIPS techniques, including hashing, tree search, and graph search. 

In 2010, Jegou et al. pioneered a novel solution in the similarity search domain by employing a divide-and-conquer strategy, which involved subdividing vectors into sub-vectors followed by quantization. This product quantization based method facilitates rapid estimation of approximate distances between vectors represented by codes, through the pre-computation of distance tables for each code. This efficient technique for approximate nearest neighbors (ANNs) quickly became a mainstream approach in similarity search, including _item-item search_ and _user-item search_. Beyond parallel quantization methods, Lian et al. have explored sequential quantization to discretize item embeddings, thereby enhancing relevance score estimation and reducing memory requirements in recommender systems.

Parallel and sequential quantization both aim to establish one-to-one mappings between vectors and code combinations, expanding horizontally and vertically, respectively, and have been validated in similarity search. However, there is currently no method that combines these approaches to finely segment and represent vectors. Additionally, similarity search techniques for weight matrices and recent low-rank adaptation (LoRA) methods share similarities in achieving approximate effects through matrix compression. In the future, these methods may also find application in parameter-efficient finetuning for recommendations, offering potential new directions for efficiency-oriented applications.

## Quality-oriented Applications

Building high-quality recommender systems is imperative to effectively cater to users' increasing information demands. 
Both academia and industry have explored various strategies to this end. These strategies include data augmentation, as demonstrated by Song et al., which entails generating synthetic data from existing datasets through techniques like item masking. Additionally, hyperparameter tuning, exemplified by Wu et al., automates the optimization of model settings, thereby mitigating the laborious process of grid search. Moreover, feature engineering, as elucidated by Schifferer, enhances feature selection and data preprocessing.

VQ enhances recommender system quality by serving as a foundational step, specifically in item indexing for generative retrieval, a process which is further detailed in discrete tokenization applications. Furthermore, VQ aligns diverse modalities with soft constraints, facilitating multimodal feature learning.

### Feature Enhancement 

Presently, recommender systems face challenges in cold-start scenarios where user interactions are sparse.
By integrating features such as item combination patterns and category information through VQ, these systems can be significantly enhanced.

To effectively utilize the data of active users, Pan et al. apply VQ to user interest clusters, facilitating cluster-level contrastive learning, which balances the personalization of representations between inactive and active users.
Their auto-quantized approach captures cluster-level similarities through VQ, in contrast to SimCLR proposed by Chen et al., which focuses solely on instance-level similarities.
To harness item combination patterns, Luo et al. propose VQA, which combines neural attention mechanism and VQ to determine the attention of candidate combination patterns.
To continuously generate and optimize the entity category trees over time, another study, CAGE enables the simultaneous learning and refinement of categorical code representations and entity embeddings in an end-to-end manner for ID-based recommendation.

However, these efforts rely on ID-based approaches, which may not be optimal in current diverse multimodal content landscape. Exploring methods to effectively leverage VQ techniques to enhance information from text, images, and other multimodal sources, and integrating it with recommendation features, presents a promising avenue for research.

### Modality Alignment

Another interesting branch of work aims to improve modality alignment in recommender systems.
Transferable recommender systems are becoming increasingly important which can quickly adapt to new domains or scenarios. 
However, ensuring the alignment of various modalities and preserving their distinct patterns throughout downstream training models remains a challenge.

Under transferable scenario, VQ can be used for loosening the binding between item _text_ and _ID_ representation, as a sparse representation technique. Hou et al. introduce VQ to represent items in a compact form, capturing the diverse characteristics of the products and addressing the transferability issues in sequential recommender systems. 
In contrast, Hu et al. employ product quantization to impose additional modality constraints, targeting the mitigation of the modality forgetting issue in two-stage sequential recommenders. This involves transforming dissected _text_ and _visual_ correlations into discrete codebook representations to enforce tighter constraints.

Hence, VQ serves as a potent semantic bridge, particularly with the rise of Large Language Models (LLMs), facilitating connectivity across diverse modalities or domains. However, existing approaches primarily focus on aligning two modalities. Addressing multimodal scenarios involving more than three modalities necessitates novel solutions.

### Discrete Tokenization

Tokenizing items and users in recommender systems has involved numerous strategies. Traditional methods often use atomic item identifiers (IDs), which can result in cold start problems. Later developments, inspired by document retrieval techniques like DSI and NCI, introduced tree IDs using multi-layer K-Means to achieve discrete yet partially shared item tokens, though semantic discrepancies remained an issue. 

To address this, one line of research applies _embedding-level reconstruction_ task. For example, Rajput et al. developed the TIGER model based on RQ-VAE, consisting of three steps: extracting item embeddings from content, discretizing these embeddings via residual quantization, and applying the discretized item tokens for sequence recommendation. Due to the inherent nature of residual quantization that can organise the tokens in a hierarchical manner, such approach proved highly successful and foundational for future research. Subsequent projects like LC-REC expanded on this by integrating item tokens into large models, hinting at the development of foundational recommendation models. Instead, some researchers_jin2023language_ optimize this process further at _text-level reconstruction_ by treating item tokenization as a translation task within an encoder-decoder-decoder framework, using standard VQ on the top of the first decoder outputs that also achieves substantial performance.

However, the exploration of multimodal and multi-domain item tokenization remains limited, and this area presents a promising opportunity for advancing foundational recommender systems.

## Future Directions

In this section, we discuss the current challenges and emerging opportunities for future research in VQ4Rec.

### Codebook Collapse Problem

There are some limitations associated with the capability of VQ. For example, the challenge of codebook collapse may arise when only a minor portion of the codebook is effectively utilized. VQ-VAE employs STE to grant differentiability to VQ, consequently, many entries in the codebook remain unused or underutilized, restricting the model capacity to accurately represent and reconstruct input data. This core issue extends its impact to subsequent developments in recommender systems employing PQ-VAE and RQ-VAE, which impairs the recommender system's ability to offer varied and personalized recommendations to users as it fails to capture the diversity of the data. At present, preliminary endeavors have yielded encouraging results, with the scholarly community being urged to continue their research efforts in this direction.

### Item Discovery

In item tokenization scenarios, the codebook space significantly exceeds the number of items in the dataset, suggesting that many potential code combinations remain untapped. Providing human-readable description for these new code combinations, especially in generative recommendation, represents a valuable direction. For instance, in product recommendations, this can help merchants develop products tailored to user demands; in video recommendations, it allows platforms to create personalized content based on the description. Currently, code training mainly relies on item embedding reconstruction tasks. A viable alternative is an end-to-end reconstruction task based on item content such as title and description, where new code combinations are inputted into the decoder to generate the corresponding item content.

### User Tokenization

Current VQ encoding schemes primarily focus on item discretization and have shown success in generative recommendation scenarios. However, discretizing user representation, i.e., user tokenization, also presents significant opportunities for research. For instance, Liu et al. has achieved substantial storage efficiency by applying discretization to both user and item in click through rate prediction. A pressing challenge is to enhance the quality of user tokens, which could enable large models to offer personalized responses through model personalization.

### Multimodal Generative Recommendation

Item semantic tokenization is currently the leading method for indexing items in generative recommender systems. However, current methods are mostly text-based, although multimodal semantic tokenization has begun to emerge in tasks such as text-to-image and video segmentation. In the big data era, leveraging multimodal features offers a more comprehensive representation of items. Therefore, the development and application of multimodal tokenization techniques in recommender systems represents a critical advancement.

### RS--LLM Alignment

The significant success of large language models has established them as foundational elements across multiple fields. Current efforts increasingly focus on aligning object features from diverse domains with LLMs, enhancing their explainability and multimodal understanding. For example, LC-Rec has successfully finetuned discretized item IDs obtained by RQ-VAE on the LLaMA model, validating this strategy in the recommendation domain. Future endeavors could involve integrating data from various domains to develop a foundational recommendation model with versatile skills.

### Codebook Quality Evaluation

In some scenarios, the process of codebook generation and the recommendation task are not executed through end-to-end training. For instance, in item tokenization, item tokens are initially derived from item semantics before being evaluated in applications like sequential recommendation. Evaluating code quality through downstream tasks is both time-consuming and resource-intensive, suggesting a need for optimization. Therefore,  the exploration of methodologies for assessing code quality through the comparison of generated tokens against original inputs represents a significant and promising research direction.

### Efficient Large-scale Recommender Systems

As large-scale models proliferate, the demand for efficient model training and inference is escalating within the recommendation community. VQ is emerging as a promising tool for enhancing the efficiency of large recommender systems, alongside other popular techniques like distillation and quantization. For instance, Lingle et al. and Wu et al. have demonstrated that optimizing the attention mechanism through VQ can achieve linear time complexity in image generation and recommendation task, respectively. However, these approaches typically involve smaller models and embedding dimensions that can be efficiently handled using a single codebook. In contrast, for larger models like LLaMA, which has embedding dimensions as large as 4096, the straightforward use of VQ may not be as effective. Exploring the integration of parallel quantization techniques with linear attention could potentially offer a viable solution.

## Conclusion

VQ has become a pivotal element in the development of innovative solutions across various scenarios in recommender systems. With the advent of large language models, there has been a notable shift towards generative recommendation methods, where residual quantization has been widely adopted for its inherent advantages. However, the research of VQ4Rec is still in its early stage. This paper offers a comprehensive overview of current research in VQ4Rec, highlighting both efficiency-oriented and quality-oriented approaches. Additionally, we identify and discuss the open challenges and potential avenues for advancement. We hope this survey will foster continued exploration and innovation in VQ4Rec.