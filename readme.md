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

## Introduction

<p align="center">
<img src="images/PaperCount.png" alt="paper_count" />
</p>

Vector quantization (VQ), a cornerstone technique in signal processing, was originally introduced by Gray and his team in the 1980s to compress data representation while preserving the fidelity of the original signal. The foundational standard VQ technique aims to compress the entire representation space into a compact codebook containing multiple codewords, typically using a single code to approximate each vector. To improve the precision of quantization, advanced methods such as product quantization~\cite{sabin1984product} and residual quantization~\cite{juang1982multiple,gray1998quantization,rq} were introduced, representing {parallel} and {sequential} approaches, respectively. These VQ techniques have proven to be highly effective in domains including speech~\cite{makhoul1985vector,abe1990voice} and image coding~\cite{nasrabadi1988image,cosman1993using}.

