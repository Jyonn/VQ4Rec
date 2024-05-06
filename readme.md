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

Vector quantization, renowned for its unparalleled feature compression capabilities, has been a prominent topic in signal processing and machine learning research for several decades and remains widely utilized today. With the emergence of large models and generative AI, vector quantization has gained popularity in recommender systems, establishing itself as a preferred solution. This paper starts with a comprehensive review of vector quantization techniques. It then explores systematic taxonomies of vector quantization methods for recommender systems (VQ4Rec), examining their applications from multiple perspectives. Further, it provides a thorough introduction to research efforts in diverse recommendation scenarios, including efficiency-oriented approaches and quality-oriented approaches. Finally, the survey analyzes the remaining challenges and anticipates future trends in VQ4Rec, including the challenges associated with the training of vector quantization, the opportunities presented by large language models, and emerging trends in multimodal recommender systems. We hope this survey can pave the way for future researchers in the recommendation community and accelerate their exploration in this promising field.

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

The standard VQ~\citep{buzo1980speech,vq} serves as the atomic component for the latter two VQ techniques. Formally, given a set of object vectors $\mathbf{E} \in \mathbb{R}^{N \times D}$, a function $f$ (e.g., $k$-means) is required to produce a codebook $\mathbf{C} \in \mathbb{R}^{K \times D}$ such that the sum of distances between all vectors in $\mathbf{E}$ and their corresponding nearest code vectors in $\mathbf{C}$ is minimized, as illustrated in Figure 2(a). We can formally express this using the following equations:

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
Due to its storage efficiency and capability for fast approximate nearest neighbor searches, product quantization has become a popular solution in the information retrieval domain, particularly for image retrieval tasks, as evidenced by several studies~\cite{cao2017deep,jang2021self,chen2022adversarial}. Nonetheless, it overlooks the potential for significant inter-correlations among sub-vectors, which may affect the quantization performance and subsequent downstream tasks.

#### Optimized Product Quantizaiton (OPQ)

To eliminate the interdependence among multiple subspaces, optimized product quantization is introduced and uses the learnable rotation matrix $\mathbf{R} \in \mathbb{R}^{D \times D}$ for automatically selecting the most effective orientation of the data in the high-dimensional space. Such rotation minimizes the interdependence among different subspaces, allowing for a more efficient and independent quantization process, which can be defined as:

$$
\displaylines{\mathbf{E}^\prime = \mathbf{E} \times \mathbf{R}, \\
\mathbf{I} = \mathbf{R}^T \times \mathbf{R},}
$$

where $\mathbf{E}^\prime$ is the rotated matrix, and $\mathbf{I}$ represents the identity matrix. Next, $\mathbf{E}^\prime$ will be operated by product quantization, as described in Sec~\ref{sec:pq}. It is important to note that the rotation matrix $\mathbf{R}$ is trained with the codebooks.
Once trained, the $i$-th original vector can be approximated by:

$$
\mathbf{e}\_i \approx \left[\cdots, \mathbf{c}^{(j)}_{x\_j}, \cdots,\right] \times \mathbf{R}^T \quad \text{for } j \in \{1, 2, \ldots, M\}.
$$

### Sequential Vector Quantization

Standard VQ and parallel VQ typically yield _rough_ approximations of vectors. Specifically, each dimension of the original vector can only be approximated by one single value from the corresponding code vector, leading to substantial information loss. 
Taking standard VQ as an example, the difference between the original vector $\mathbf{e}$ and its corresponding code $\mathbf{c}$, denoted by $\mathbf{e}-\mathbf{c}$, reflects the unique characteristics that cannot be represented by $\mathbf{c}$.

To achieve a more _precise_ quantization, approaches like residual quantization~\cite{juang1982multiple,rq} and additive quantization~\cite{aq} have been developed, falling under the umbrella of sequential quantization. This method employs multiple codebooks, with each codebook approximates every dimension of the original vectors. Essentially, every codebook offers a distinct approximation perspective of the vectors, and the accuracy of these approximations improves with an increase in the number of codebooks. As illustrated in Figure 2, using the first layer codebook approximates `0.3` (the first dimension of the original vector) as `0.5` (the first dimension of the code vector in the first codebook). After applying the second codebook, it is more accurately approximated as `0.5 + (-0.3) = 0.2` (the first dimension of the code vector in the second codebook).

#### Residual Quantization (RQ)

By designing $M$ individual codebooks where, as depicted in Table 1, code vectors have the full same length of the input vector, residual quantization aims to approximate the target vectors by compressing their information in a coarse-to-fine manner. Specifically, the codebooks are learned iteratively from the residual representations of the vectors. This process can be formulated as: $\mathbf{E}^{(j+1)} = \mathbf{E}^{(j)} - \mathbf{X}^{(j)}\mathbf{C}^{(j)}$, where $\mathbf{E}^1 = \mathbf{E}$, $\mathbf{C}^{(j)}$ is the $j$-th codebook with size $K\_j$, and $\mathbf{X}^{(j)} \in \mathrm{\{0, 1\}}^{N}$ is a one-hot mapper, where $\mathbf{X}^{(j)}\_{i,k}=1$ only if the $k$-th code is the nearest to the $i$-th vector of $\mathbf{E}^{(j)}$ in the codebook $\mathbf{C}^{(j)}$. After iteratively residual approximation, the $i$-th original vector can be represented by:

$$
\displaylines{\mathbf{e}\_i \approx \sum_j^M \mathbf{c}^{(j)}\_{x\_j}, \\
\textit{where }x\_j = \underset{k}{\text{argmin}}\,\mathbf{X}^{(j)}\_{i,k}.}
$$

It is important to note that, as $M$ increases, the approximated representation tends to be finer.

#### Additive Quantization (AQ)

Similar to residual quantization, additive quantization aims to approximate the target vectors by aggregating one selected code per codebook. However, residual quantization employs a greedy approach by selecting only the _nearest_ neighbor (i.e., $\mathbf{c}^{(j)}\_{x\_j}$) within the current (i.e., $j$-th) layer, which does not guarantee the global optimum. Instead, codebooks here are sequentially learned using beam search, where top candidate code combinations (_not the only one_) from the first $j$ codebooks are selected to infer the $(j+1)$-th codebook. Hence, the $i$-th original vector can be approximated just as that in residual quantization.