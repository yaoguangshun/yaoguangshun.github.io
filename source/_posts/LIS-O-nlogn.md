---
title: LIS的O(nlogn)实现原理
date: 2017-08-30 16:07:39
tags: [dp]
mathjax: true
categories: 算法
---
　　朴素的LIS的**O(n²)**算法是用dp，用d[i]表示a[i]中以i为结尾的LIS的值，那么状态转移方程可表示为**d[i] = max{d[j] | j < i && a[j] < a[i]} + 1**。显然，对于一个i下的两个不同决策**j，k (j,k < i)，若a[j] < a[k]，d[j] >= d[k]，而a[i] > a[k]时，k显然没有j决策更优**。通过这种思想，我们发现决策有一定的单调性，从而其中可以用二分查找来降低时间复杂度。
## 实现原理
　　我们可以维护一个决策序列B，用**B[i]**表示子序列长度为i时的最小末尾(a中的值)。可以看出如果**i > j**，一定有**B[i] >= B[j]**。原因很简单，看B的定义就可以知道。由上可知B是单调的序列，对于其中的的值可以进行二分查找。

　　所以可以按顺序枚举**a[i]**，如果**a[i]**的值比B中的最大值要大，则将**a[i]**放入**B[i]**的尾部，最大长度加1; 否则对B进行二分查找以找到**a[i]**该放入的位置，**pos = min{j | B[j] > a[i]}**，用a[i]代替B[pos]，使a[i]成为子序列长度为pos时的最小值。大致过程如下：
```c++
for (int i=1; i<n; i++) {
	if (a[i] > B[len-1]) {
		B[len] = a[i];
		len++;
	}
	else {
		pos = binarySearch(len, a[i]);
		B[pos] = a[i];
	}
}
```
完整代码请戳：[ljm's github](https://github.com/mingming97/Algorithms/blob/master/c%2B%2B/LIS-nlogn.cpp)