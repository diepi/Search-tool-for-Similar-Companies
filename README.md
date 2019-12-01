# Search-tool-for-Similar-Companies
The project was created by University of Manchester and PEAK AI Manchester.

## Introduction
Nowadays many companies spend a great amount of time on searching for similar business for market analysis or investment purposes. It was until recently, when it was proposed to use machine learning and deep learning techniques to implement this process in easier and more efficient way.

This particular project focuses on scraping the textual data from all the official financial website such as FT, Telegraph or London Stock Exchange. Then we used deep learning technique to find similar companies to a particular company (search 1) and used machine learning technique to find the similar companies within an industry (search 2).

For search 1 we have decided to use the Doc2vec embeddings, which is famous for paragraph embedding. For a second search, we have used the machine learning technique TFIDF to find the similar companies within an industry, which is predefined by our dictionaries. The process of second search is shown in the following picture.

<img src="tfidf.pdf" width="50%"/>

## Getting Started
The code attached in this file contain the code for search tool including search 1 and search 2. The code is written in python and is dependant on libraries:

```
numpy
nltk
gensim
```

## Running the Codes
```
main.py
```
Here we can modify the learning rate, epoch and batch size to train the simple MLP. Moreover, the trained weights will be stored in the user defined file, which will be later used for test and prediction.

```
=== Epoch: 0/10 === Iter:500 === Loss: 2.39 === BAcc: 0.10 === TAcc: 0.10 === Remain: 12 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:1000 === Loss: 2.35 === BAcc: 0.13 === TAcc: 0.11 === Remain: 12 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:1500 === Loss: 2.35 === BAcc: 0.12 === TAcc: 0.11 === Remain: 11 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:2000 === Loss: 2.33 === BAcc: 0.11 === TAcc: 0.11 === Remain: 12 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:2500 === Loss: 2.30 === BAcc: 0.12 === TAcc: 0.11 === Remain: 11 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:3000 === Loss: 2.30 === BAcc: 0.12 === TAcc: 0.11 === Remain: 11 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:3500 === Loss: 2.27 === BAcc: 0.14 === TAcc: 0.12 === Remain: 11 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:4000 === Loss: 2.26 === BAcc: 0.16 === TAcc: 0.12 === Remain: 10 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:4500 === Loss: 2.24 === BAcc: 0.19 === TAcc: 0.13 === Remain: 10 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:5000 === Loss: 2.24 === BAcc: 0.19 === TAcc: 0.14 === Remain: 10 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:5500 === Loss: 2.21 === BAcc: 0.24 === TAcc: 0.15 === Remain: 10 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:6000 === Loss: 2.20 === BAcc: 0.20 === TAcc: 0.15 === Remain: 10 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:6500 === Loss: 2.16 === BAcc: 0.27 === TAcc: 0.16 === Remain: 10 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:7000 === Loss: 2.16 === BAcc: 0.27 === TAcc: 0.17 === Remain: 10 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:7500 === Loss: 2.18 === BAcc: 0.25 === TAcc: 0.17 === Remain: 11 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:8000 === Loss: 2.17 === BAcc: 0.26 === TAcc: 0.18 === Remain: 10 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:8500 === Loss: 2.15 === BAcc: 0.27 === TAcc: 0.18 === Remain: 10 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:9000 === Loss: 2.12 === BAcc: 0.25 === TAcc: 0.19 === Remain: 10 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:9500 === Loss: 2.11 === BAcc: 0.31 === TAcc: 0.19 === Remain: 10 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:10000 === Loss: 2.09 === BAcc: 0.30 === TAcc: 0.20 === Remain: 10 Hrs 0 Mins 0 Secs ===
=== Epoch: 0/10 === Iter:10500 === Loss: 2.09 === BAcc: 0.33 === TAcc: 0.21 === Remain: 10 Hrs 0 Mins 0
```

## Results
* learning rate: 0.001
* batch size: 500
* number of particles for ensemble: 10

### Training plot
<img src="alldnn1.png" width="50%"/>

### Test accuracy
SGD: 0.8922;
NAG: 0.936;
Adam: 0.9699;
IEnK: 0.886.

