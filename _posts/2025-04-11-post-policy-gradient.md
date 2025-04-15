---
title: "Policy Gradient"
mathjax: true
classes: wide
excerpt_separator: "<!--more-->"
categories:
  - Blog
tags:
  - RL
---

## Background
Policy gradient is one of the methods in reinforcement learning that directly computes the gradient of the objective function and uses it to update our policy. In reinforcement learning, let's say we have the following trajectory distributions over states and actions:

\\[
    p_{\theta}(\tau)=p(s_1,a_1,...,s_T,a_T)=p(s_1) \prod_{t=1}^{T} \pi_{\theta}(a_t \mid s_t)p(s_{t+1} \mid a_t,s_t)
\\]

If we have a reward function $r(s_t, a_t)$, our objective of reinforcement learning can be written as the following:

\\[
    \theta^{*}=\underset{\theta}{\operatorname{argmax}}E_{\tau \sim p_{\theta}(\tau)}[\sum_{t}r(s_t,a_t)]=\underset{\theta}{\operatorname{argmax}}J(\theta)
\\]

which we want to maximize the expectation of total rewards at each timestep $t$, and the trajectory distribution depends on our policy $\pi_{\theta}$, initial state probability $p(s_1)$ and transition probability $p(s_{t+1} \mid a_t, s_t)$


## Direct policy differentiation
For simplicity, let's call the sum of the reward function to be $r(\tau)$, then we can write our objective function as the following:
\\[
    \displaylines{
        J(\theta)=E_{\tau \sim p_{\theta}(\tau)}[r(\tau)]=\int p(\tau)r(\tau)d\tau \cr
        \nabla_{\theta} J(\theta)=\int \nabla_{\theta} p(\tau)r(\tau)d\tau
    }
\\]

However, this notation does not seem to be very practical for taking the gradient because we still have initial state probability distribution and transition probability in our expression which are unknown. To simplify the expression above, we can use one convenient identity:
\\[
    \nabla_{\theta}log(p(\tau))= \frac{\nabla_{\theta}p_{\theta}(\tau)}{p(\tau)}
\\]

With the identity above, we can rewrite $J(\theta)$ to be:

\\[
    \nabla_{\theta} J(\theta)=\int p(\tau)\nabla_{\theta}log(p_{\theta}(\tau))r(\tau)d\tau = E_{\tau \sim p_{\theta}(\tau)}[\nabla_{\theta}log(p_{\theta}(\tau))r(\tau)]
\\]

The, we can plug in our trajectory distribution to $\nabla_{\theta}logp_{\theta}(\tau)$:

\\[
    \nabla_{\theta}logp_{\theta}(\tau)=\nabla_{\theta}[\cancel{logp(s_1)} + \sum_{t}log\pi_{\theta}(a_t \mid s_t) + \cancel{logp(s_{t+1} \mid a_t, s_t)}]
\\]

Since the initial state distribution and transition probability are independent of $\theta$, those will result in 0 when we take the gradient. The gradient will become the following:

\\[
    \nabla_{\theta} J(\theta)=E_{\tau \sim p_{\theta}(\tau)}[(\sum_{t=1}^{T}\nabla_{\theta}log\pi_{\theta}(a_t \mid s_t))(\sum_{t=1}^{T}r(s_t,a_t))]
\\]
To approximate the expectation, we can sample *N* times and take the average:
\\[
    \nabla_{\theta} J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}(\sum_{t=1}^{T}\nabla_{\theta}log\pi_{\theta}(a_{i,t} \mid s_{i,t}))(\sum_{t=1}^{T}r(s_{i,t},a_{i,t}))
\\]

With this expression, we can simply generate $N$ samples, calculate the sum of rewards, and then update our parameters $\theta$ as $\theta \leftarrow \theta + \alpha \nabla_{\theta}J(\theta)$