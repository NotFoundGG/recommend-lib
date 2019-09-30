<!--
 * @Author: Yu Di
 * @Date: 2019-09-29 10:33:55
 * @LastEditors: Yudi
 * @LastEditTime: 2019-09-30 14:42:02
 * @Company: Cardinal Operation
 * @Email: yudi@shanshu.ai
 * @Description: 
 -->

# fair-comparison-for-recommendation

1. Item-Ranking recommendation algorithms reimplementation with pytorch-gpu and [Surprise](https://github.com/NicolasHug/Surprise) Toolkit.

## The requirements are as follows:

* python==3.7
* pandas==0.24.2
* numpy==1.16.2
* pytorch==1.0.1
* surprise==1.1.0

## Examples to run:

Default set top-K number to 10

```
python MostPopRecommender.py
python ItemKNNRecommender.py
python UserKNNRecommnder.py
python BPRMFRecommender.py --factor_num=16
python NCFRecommnder.py --batch_size=256 --lr=0.001 --factor_num=16
```
