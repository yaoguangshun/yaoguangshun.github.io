---
title: 中文分词
date: 2018-5-17 17:40:01
tags: [NLP]
mathjax: true
categories: NLP
---
##基本思路

&emsp;&emsp;中文分词其实有很多种思路，大多都是建立在HMM模型的基础上。先简要介绍一下HMM模型，HMM模型中有三个要素：A是状态转移概率分布矩阵，简单说就是在任一时刻从一个隐含状态到另一个隐含状态的转移概率构成的矩阵；B是观测概率分布矩阵，其实就是在任一时刻给定隐含状态s生成观测状态o的条件概率$P(o|s)$构成的矩阵；$\pi$是初始概率矩阵，也就是在初始状态下各隐含状态的概率。而一般的HMM模型有三个基本问题：1. 给定模型$\lambda = (A, B, \pi)$和观测序列$O = \{o_1, o_2, \dots, o_t\}$，计算$P(O|\lambda)$，这是评估问题。2. 给定观测序列$O = \{o_1, o_2, \dots, o_t\}$，求解模型$\lambda = (A, B, \pi$，使得$P(O|\lambda)$尽可能大，这是学习问题，若给定隐含状态序列S可以考虑用maximum likelihood来解决，若隐含状态序列则可以用Baum-Welch算法解决，不过这并不是本文重点。3. 给定 给定模型$\lambda = (A, B, \pi)$和观测序列$O = \{o_1, o_2, \dots, o_t\}$，求使得$P(S|O)$最大的隐含状态序列$S = \{s_1, s_2, \dots, s_t\}$，这被称为解码问题或预测问题。对于分词这个任务来说，主要涉及到的是第三个问题。
&emsp;&emsp;jieba分词的源码就提供了解决这个问题的一个很好的范例。将隐含状态集合定义为$\{S, B, M, E\}$，S的含义是单字，B的含义是词头，M的含义是词中，E的含义是词尾。在 `jieba/finaseg/prob_start.py`中定义了初始概率$\pi$，在 `jieba/finaseg/prob_trans.py`中定义了状态转移概率$A$，在`jieba/finaseg/prob_emit.py`中定义了状态观测概率分布$B$，在用基于统计的方法获得以上这些之后就用Viterbi算法求一条使得$P(S|O)$最大的路径就好。（关于Viterbi算法还是在另篇文章中再说。）
&emsp;&emsp;考虑一下，此时如果我们不知道B，该如何定义要求解的函数。可以试着模仿Viterbi的想法，用$\delta_i(s)$表示到第i个字时状态为s时的最优值，则$\delta_{i+1}(s') = max\{\delta_i(s)a_{ss'}P(s|i),  s\in\{S, B, M, E\}\}$，其中$a_{ss'}$是转移概率，$P(s|i)$表示第i个字状态是s的概率（这样定义是有着一定数学原理的，具体推导也借鉴了Viterbi算法原本的定义，核心思想是极大似然）。转移概率可通过统计的方法得到，那么$P(s|i)$呢？影响到这个概率的因素很多，不妨将这个问题转化为一个seq2seq的问题，输入一个序列，输出各个位置的4-tag标注。中文中通过前后文语境都能作为序列标注的依据，从而考虑使用Bi-directional的LSTM来进行这个任务。只要将输出接一层softmax就可以将结果当作概率使用。



## 数据处理

&emsp;&emsp;有一个经典的亚研院的语料库就是4-tag标注的，大概长这个样子。

![msrtrain.png](https://i.loli.net/2019/01/13/5c3af89cb9124.png)

&emsp;&emsp;首先要明确我们训练数据的lstm的输入，应该是batch_size * sentence_len的一个tensor。以标点符号为分隔，汉语中单句话一般没有太长，所以此处统一每个句向量的长度为32。获得句向量的方式很简单，将文章中所有出现的字做成一个字典，将每个字用其在字典中对应的下位置表示，不足的长度用0补齐，就得到了句向量的表示。而对于tag，则可以使用one-hot的编码，注意要为补足句向量的0留一个编码位置，所以一共有5类tag。因此一句话中tag为s的字将被编码为[1, 0, 0, 0, 0]，tag为b的将被编码为[0, 1, 0, 0, 0]，以此类推。
&emsp;&emsp;明确了以上过程，就可以开始细节上的处理。注意到每行前面有很多“/s，”/s，‘/s，’/s一类的东西，我们的数据是不需要这类东西的，可以用正则处理掉。处理掉之后单看每行，可以方便的使用python的re.findall提取出来一个字和标签的元组组成的列表。再将二者分别处理，分别得到句子和标签。
&emsp;&emsp;通过标签得到独热编码的过程值得记录一下。面对的问题就是给出了一个类别的列表，如何得到独热编码。可以利用numpy的花式索引方便进行。代码如下：（虽然花式索引返回的是数据的拷贝，但是使用花式索引进行赋值却是在原数组上进行操作的）
```python
data = np.array(data)	# 获得numpy的数组
one_hot = np.zeros((len(data), 5), dtype=np.float64)	# 获得len(data)个长度为5的0向量
one_hot[range(len(data)), data] = 1		# 二维花式索引 可以实现任意位置的操作
default = np.array([0,0,0,0,1], dtype=np.float64)	# 定义默认向量
default = np.tile(default, (32 - len(one_hot), 1))	# 将默认向量在列方向上重复32-len(one_hot)次
one_hot = np.concatenate((one_hot, default), axis = 0)	# 将默认部分与有效部分拼接保证长度与句向量相同
```



## 模型定义及训练

&emsp;&emsp;接下类就是定义一个双向LSTM的过程，为了简单起见，使用了keras进行，keras中还有embedding层，正好适用于我们生成的句向量。keras封装度比较高，所以代码较短，如下：

```python
inputs = Input(shape=(sent_len,), dtype='int32')
embedding = Embedding(len(charsets) + 1, word_size, input_length=sent_len, mask_zero=True)(inputs)
bilstm_layer = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedding)
output = TimeDistributed(Dense(5, activation='softmax'))(bilstm_layer)
model = Model(input=inputs, output=output) model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
json.dump(model.to_json(), open(os.path.join(filedir, model_name), "w"))
model.fit(x, y, batch_size=batch_size, epochs=epochs)
model.save_weights(os.path.join(filedir, model_weights_name))
```
&emsp;&emsp;训练好了模型，输入一句话，就能给出各个位置对应各种标签的概率。
```python
sentence_embeddings, all_sentences, sentence_len = data.get_sent_embeddings(charsets, test_sentence)
result = model.predict(sentence_embeddings, verbose=False)
```



## 计算概率最大路径

&emsp;&emsp;根据之前对$\delta_i(s)$的定义，使用dp就可以计算概率最大的路径了。此处转移概率用的是相等概率，dp算法的具体实现上利用了python字典的性质，代码如下：

```python
def viterbi(nodes):
    path = {'b': nodes[0]['b'], 's': nodes[0]['s']}
    for layer_num in range(1, len(nodes)):
        old_path = path.copy()
        path = {}
        for new_tag in nodes[layer_num].keys():
            tmp = {}
            if layer_num == len(nodes) - 1:
                if new_tag in ["m", "b"]:
                    continue
            for old_path_tag in old_path.keys():
                if old_path_tag[-1]+new_tag in transpose_matrix.keys():
                    tmp[old_path_tag+new_tag] = old_path[old_path_tag] + \
                                                nodes[layer_num][new_tag] + \
                                                transpose_matrix[old_path_tag[-1]+new_tag]
            k = np.argmax(list(tmp.values()))
            path[list(tmp.keys())[k]] = list(tmp.values())[k]
    return list(path.keys())[np.argmax(list(path.values()))]
```



## 总结

通过以上几个步骤，就完成了对一句话4-tag的标注。自己试了几句话，效果还不错，见图：

![result1.png](https://i.loli.net/2019/01/13/5c3af89cb53f3.png)

![result2.png](https://i.loli.net/2019/01/13/5c3af89cb71fb.png)

完整代码可以参照我的[Github](https://github.com/mingming97/Chinese-Word-Split)。

