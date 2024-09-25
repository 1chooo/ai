# [CIFAR-10 Python](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python)

Author: [1chooo](https://1chooo.com)

## With DNN

### My gained knowledge

I have tested a lot of experiment to improve the model; however, all of the results still surround to about forty percent accuracy. 

#### Below are the experiments I have conducted:
* `epoches: from 20 -> 14 -> 9 -> 14`.
* `batch_size: from 100 -> 500`.
* `optimizer: adam and rmsprop`.

Even though I designed a lot of experiments, the accuracy did not increase significantly. I have considered the reasons, and here are my conclusions. 

First, our Deep Neural-Network model was limited by the size of the CIFAR-10 dataset, which consisted of up to 50000 training_data and up to 10000 testing_data. The larger datasets made it difficult for the DNN model to capture all the necessary values during training, which resulted in less accuracy even when we changed several variables. 

Second, given the large amount of data, I could have tried to drop out the data that affected the results. However, I thought that we might be able to choose the Convolutional Neural-Network instead because it was more suitable for dropping out the worse neural in our model.

In conclusion, I am excited to have the opportunity to improve my deep-learning skills with this dataset and to review what I have learned before.

### Reference

* [Day 20 ~ AI從入門到放棄 - 新的資料集](https://ithelp.ithome.com.tw/articles/10248873)
* [簡單使用keras 架構深度學習神經網路 — 以cifar10為例](https://medium.com/@a227799770055/%E7%B0%A1%E5%96%AE%E4%BD%BF%E7%94%A8keras-%E6%9E%B6%E6%A7%8B%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-%E4%BB%A5cifar10%E7%82%BA%E4%BE%8B-b8921ca239cf)

