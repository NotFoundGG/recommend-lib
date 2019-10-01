<!--
 * @Author: Yu Di
 * @Date: 2019-09-29 10:33:55
 * @LastEditors: Yudi
 * @LastEditTime: 2019-10-01 19:10:16
 * @Company: Cardinal Operation
 * @Email: yudi@shanshu.ai
 * @Description: 
 -->

# fair-comparison-for-recommendation

1. you can also download experiment data from links below: 
    - [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)

2. Item-Ranking recommendation algorithms reimplementation with pytorch-gpu and [Surprise](https://github.com/NicolasHug/Surprise) Toolkit.

## The requirements are as follows:

* python==3.7
* pandas>=0.24.2
* numpy>=1.16.2
* pytorch>=1.0.1
* sklearn>=0.21.3
* surprise==1.1.0
* deepctr-torch==0.1.2

## Examples to run:

Default set top-K number to 10

```
python MostPopRecommender.py
python ItemKNNRecommender.py
python UserKNNRecommnder.py
python BPRMFRecommender.py --factor_num=16
python NCFRecommnder.py --batch_size=256 --factor_num=16 --model_name=NeuMF-pre
python DeepFMRecommender.py
```
