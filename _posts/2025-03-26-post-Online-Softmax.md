---
title: "Online Softmax"
mathjax: true
excerpt_separator: "<!--more-->"
categories:
  - Blog
tags:
  - LLM
---

This post shows the basic derivation from the traditional softmax, to safe softmax and online safe softmax. The idea was first proposed by the engineers from Nvidia [[1]](#1).

## Softmax
The softmax function is defined as the following:
\\[
  y_i=\frac{e^{x_i}}{\sum\limits_{j=1}^{V}e^{x_j}}
\\]

It is obvious that we need to iterate over the input vector twice:
1. Scan the first time to calculate the sum of $e^{x_j}$ (1 load)
2. Scan the second time to calculate $y_i$ for each element (1 load + 1 store)  

Therefore, in total we will need 2 loads + 1 store operation per vector element. However, this naive implementation has the drawback of overflow and underflow when $x_j$ gets extreme.

## Safe softmax
Safe softmax was proposed by simply subtracting the maximum value from the input vector in the exponent.
\\[
  y_i=\frac{e^{x_i-\max\limits_{k=1}^{V}x_k}}{\sum\limits_{j=1}^{V}e^{x_j-\max\limits_{k=1}^{V}x_k}}
\\]
In addition to softmax, we need to add an extra iteration to find the maximum value from the input vector. This will add an extra load operation per vector element. Is there any way we can do better than this and still have the advantage of safe softmax?

## Online safe softmax
Yes, the author from [[1]](#1) suggested an intricate way of calculating safe softmax in an online manner which the normalizer $\sum\limits_{j=1}^{V}e^{x_j-\max\limits_{k=1}^{V}x_k}$ can be calculated within one loop.

At each step *S*, the sum of the normalizer can be written as the following:
\\[
  \sum_{j=1}^{S}e^{x_j-m_S}
\\]
where $m_S$ is the maximum until step $S$. And now we can split out expression as step $S-1$ and step $S$:
\\[
  \displaylines{
    \sum_{j=1}^{S}e^{x_j-m_S} \cr
    =\sum_{j=1}^{S-1}e^{x_j-m_S} + e^{x_S-m_S}
  }
\\]

However, at step $S-1$, we do not have $m_S$ but $m_{S-1}$, we need to substitute $m_{S-1}$ into our equation.

\\[
  \displaylines{
    =\sum_{j=1}^{S-1}e^{x_j-m_{S-1}}e^{m_{S-1}-m_S} + e^{x_S-m_S} \cr
    =d_{j-1}e^{m_{S-1}-m_S} + e^{x_S-m_S}
  }
\\]

From the above, we show that online safe softmax normalizer is equivalent to the safe softmax normalizer.

With online safe softmax, it reduces the iteration from 3 times to 2 times comparing to offline safe softmax. It also helps to reduce memory access to 2 loads + 1 store operations, which is the same as naive softmax.

The idea of online softmax is one of the fundamentals of Flash Attention [[2]](#2) which we will discuss in future post.


## Reference
<a id="1">[1]</a>
Milakov, M., & Gimelshein, N. (2018). Online normalizer calculation for softmax. arXiv preprint arXiv:1805.02867.

<a id="2">[2]</a>
Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35, 16344-16359.
