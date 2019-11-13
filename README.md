<!--
 * @Author: Yu Di
 * @Date: 2019-09-29 10:33:55
 * @LastEditors: Yudi
 * @LastEditTime: 2019-11-12 17:03:46
 * @Company: Cardinal Operation
 * @Email: yudi@shanshu.ai
 * @Description: 
 -->

# fair-comparison-for-recommendation

1. you can also download experiment data from links below: 
    - [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
    - [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
    - [MovieLens 10M](https://grouplens.org/datasets/movielens/10m/)
    - [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)
    - [Netflix Prize Data](https://archive.org/download/nf_prize_dataset.tar)
    - [CiteULike](https://github.com/js05212/citeulike-a)
    - [Amazon](http://jmcauley.ucsd.edu/data/amazon/links.html)
    - [Yelp]()

2. Item-Ranking recommendation algorithms reimplementation with pytorch, numpy and scipy.

3. Before running, you need first run `python setup.py build_ext --inplace` in directory `util` to generate `.so` file for `macOS` or `.pyd` file for `WindowsOS` used for further import.

## The requirements are as follows:

* python==3.7
* pandas>=0.24.2
* numpy>=1.16.2
* pytorch>=1.0.1
* sklearn>=0.21.3
* Cython>=0.29.12

## Overview for all algorithms

| Algo. | File | Refer. |
| ------ | ------ | ------ |
| Popular | MostPopRecommender.py | - |
| Item-KNN | ItemKNNRecommender.py | - |
| User-KNN | UserKNNRecommender.py | - |
| SLiM | SLiMRecommender.py | SLIM: Sparse Linear Methods for Top-N Recommender Systems |
| MF | MFRecommender.py | Probabilistic matrix factorization |
| PureSVD | PureSVDRecommender.py | Performance of recommender algorithms on top-N recommendation tasks |
| SVD++ | SVDppRecommender.py | Factorization meets the neighborhood: a multifaceted collaborative filtering model |
| WRMF | WRMFRecommender.py | Collaborative Filtering for Implicit Feedback Datasets |
| BPR-MF | BPRMFRecommender.py | BPR: Bayesian Personalized Ranking from Implicit Feedback |
| FM | NFMRecommender.py | Factorization Machines |
| NeuFM | NFMRecommender.py | Neural Factorization Machines for Sparse Predictive Analytics |
| NeuMF | NCFRecommender.py | Neural Collaborative Filtering |
| Item2Vec |  | Item2Vec: Neural Item Embedding for Collaborative Filtering  |
| AutoRec |  | AutoRec: Autoencoders Meet Collaborative Filtering |
<!-- | eALS |  | https://arxiv.org/pdf/1708.05024.pdf | -->


## Examples to run:

Default set top-K number to 10, you can change top-K number by modifying `topk` argument.

```
python MostPopRecommender.py
python ItemKNNRecommender.py --sim_method=pearson
python UserKNNRecommnder.py --sim_method=jaccard
python SLiMRecommender.py
python MFRecommender.py --biased=False
python PureSVDRecommender.py --factors=150
python SVDppRecommender.py
python WRMFRecommender.py
python BPRMFRecommender.py --factor_num=16
python NCFRecommender.py --batch_size=256 --factor_num=16
python NFMRecommender.py --batch_size=128 --lr=0.05 --model=FM
```

Help message will give you more detail description for arguments, For example:

```
python NFMRecommender.py --help
```

## Implementation detail

- you need to add corresponding dataset file into **data** folder
- make sure you have a **CUDA** enviroment to accelarate since these deep-learning models could be based on it.

<!-- ## Simple Result Achieved for quick look

| Algo | HR@10 | NDCG@10 | MAP@10 |
| ------ | ------ | ------ | -- |
| Pop | 0.101  | 0.338 | 0.040 |
| UserKNN | 0.141  | 0.341 | 0.069 |
| ItemKNN | 0.153  | 0.351 | 0.079 |
| SLiM | 0.359 | 0.706 | 0.262 |
| NMF | 0.157 | 0.353 | 0.078 |
| PureSVD | 0.347 | 0.638 | 0.248 |
| SVD | 0.164 | 0.365 | 0.087 |
| SVD++ | 0.152 | 0.360 | 0.077 |
| WRMF | 0.586 | 0.833 | 0.451 |
| BPR-MF | 0.705 | 0.407 | 0.315 |
| NeuMF | 0.698  | 0.401 | 0.310 |
| FM | 0.209 | 0.451 | 0.119 | -->
<!-- | NeuFM(deprecated) | 0.214  | 0.453 | 0.119 | -->
