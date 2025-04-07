---
title: "Why behavioral cloning is not enough"
mathjax: true
classes: wide
excerpt_separator: "<!--more-->"
categories:
  - Blog
tags:
  - RL
---
## Background
One of the basic ideas of imitation learning is through behavioral cloning. However, behavioral cloning is not guaranteed to work well. The behavioral cloning problem in essence is different from the supervised learning problem that we are familiar with. One obvious difference is that in behavioral cloning, the data is not Independent and Identically Distributed (IID). The next action will be dependent on the current state and action.

For a supervised learning problem trained under a squence of observation $p_{data}(o_t)$, our goal would be the following by maximum likelihood estimation:
\\[
  \underset{\theta}{\operatorname{argmax}}E_{o_t \sim\ p_{data}(o_t)}[log\pi_{\theta}(a_t|o_t)]
\\]

If we train this policy $\pi_{\theta}$, the observation is sampled from the "expert". But in the test case, the obersavation is actually sampled from $p_{\pi_{\theta}}(o_t)$. Therefore, this will induce the distributional shift problem.


## Mathematical proof
Let's assume we have a trained policy $\pi_{\theta}$ and the probability of making an error is $\epsilon$, it can be written as:

\\[
  \pi_{\theta}(a \neq \pi^{*}(s)|s) \leq \epsilon
\\]

where $\pi^{*}(s)$ is the policy from the "expert" with state $s$, the state $s$ here is interchangeable with the observation $o$ above in this case.

Generally, we can assume that $p_{train}(s) \neq p_{\theta}(s)$, then we can describe the distribution of state $s$ at timestep $t$:

\\[
  p_{\theta}(s_t)=(1-\epsilon)^{t}p_{train}(s_t)+(1-(1-\epsilon)^{t})p_{mistake}(s_t)
\\]

If we start sampling from train distribution, at timestep $t$, the probability of not making any mistake woud be $(1-\epsilon)^{t}p_{train}(s_t)$, $p_{mistake}$ is an unknown distribution. Since $\epsilon$ is very small, the equation will be dominant by the first half.

To illustrate why behavioral cloning suffers from the distributional shift problem, let's write out the total variance divergence of $p_{\theta}$ and $p_{train}$:

\\[
  \displaylines{
    |p_{\theta}(s_t) - p_{train}(s_t)| = (1-\epsilon)^{t}p_{train}(s_t)+(1-(1-\epsilon)^{t})p_{mistake}(s_t) - p_{train}(s_t) \cr
    |p_{\theta}(s_t) - p_{train}(s_t)| = ((1-\epsilon)^{t}-1)p_{train}(s_t)+(1-(1-\epsilon)^{t})p_{mistake}(s_t) \cr
    |p_{\theta}(s_t) - p_{train}(s_t)| = (1-(1-\epsilon)^{t})|p_{mistake}(s_t) - p_{train}(s_t)|
  }
\\]

We know that the maximum total variance distance of two distributions is 2. Intuitively, at the first state, distribution A is 1 while distribution B is 0. At the second, distribution A is 0 while distribution B is 1. The sum of the difference will be 2.

Therefore,
\\[
  \displaylines{
    |p_{\theta}(s_t) - p_{train}(s_t)| \leq 2(1-(1-\epsilon)^t) \cr
    |p_{\theta}(s_t) - p_{train}(s_t)| \leq 2\epsilon t
  }  
\\]

The above inequality holds because for $\epsilon \in [0, 1]$, we always have $(1-\epsilon)^t \leq 1-\epsilon t$.

What we want to minimize is the expectation of the cost function:

\\[
  \displaylines{
    \sum_{t}E_{p_{\theta}(s_t)}[c_t] \leq \sum_{t} \sum_{s_t}p_{\theta}(s_t)c_t(s_t) \leq \sum_{t} \sum_{s_t} p_{train}(s_t)c_t(s_t) + |p_{\theta}(s_t) - p_{train}(s_t)|c_{max} \cr
    \sum_{t}E_{p_{\theta}(s_t)}[c_t] \leq \sum_{t} \epsilon + 2\epsilon t \cr
    \sum_{t}E_{p_{\theta}(s_t)}[c_t] \leq \epsilon T + 2\epsilon T^2
  }  
\\]

From the above, we prove that the expectation of the cost function will be subject to $O(\epsilon T^2)$. Therefore, the naive behavioral cloning that just maximize the likelihood of the expert trajectory will be extremely prone to the increase of the timestep.

## **DAgger**: Dataset Aggregation
DAgger [[1]](#1) is an iterative algorithm that trying to solve the distributional shift problem. The algorithm works as the following:

1.train $\pi_{\theta} (a_{t}|o_{t})$ from human data $D = \{o_1, a_1,...,o_N,a_N\}$  
2.run $ \pi_{\theta} (a_{t}|o_{t}) $ \text{to get data} $ D_{\pi}=\{o_1,...,o_M\}$  
3.human to label $D_{\pi}$ to get actions $a_t$  
4.Aggregate: $D \leftarrow D \bigcup D_{\pi}$  

After many iterations, we can expect that we are trained under $p_{\pi_{\theta}}(o_t)$, that eventually $p_{data}(o_t)=p_{\pi_{\theta}}(o_t)$. Although theoretically this can help to mitigate the distributional shift issue, it can suffer from step 3 where human labeling is needed. It could be difficult to acquire vast amount of human-labeled data or the labeling task itself can be extremely difficult for human to perform (e.g. label humanoid actions with many degree-of-freedoms).

That's why there many other types of reinforcement learning methods for different tasks, which we will discuss in future posts.



## Reference
<a id="1">[1]</a>
Ross, S., Gordon, G., & Bagnell, D. (2011, June). A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics (pp. 627-635). JMLR Workshop and Conference Proceedings.

