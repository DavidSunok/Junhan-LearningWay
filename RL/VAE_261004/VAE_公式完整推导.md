# VAE_261004：完整公式推导

这份文件只做一件事：把主笔记中为了直觉而省略的数学步骤完整写出来。

---

## 0. 记号与假设

单个观测为 $x$，潜变量为 $z$。

生成模型：

$$
p_\theta(x,z)=p(z)p_\theta(x\mid z).
$$

近似推断模型：

$$
q_\phi(z\mid x).
$$

标准 VAE 常用：

$$
p(z)=\mathcal N(0,I),
$$

以及对角高斯：

$$
q_\phi(z\mid x)
=
\mathcal N(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))).
$$

我们的真正最大似然目标是：

$$
\max_{\theta}\sum_{i=1}^N \log p_\theta(x_i).
$$

其中：

$$
p_\theta(x)=\int p_\theta(x,z)\,dz
=\int p(z)p_\theta(x\mid z)\,dz.
$$

---

## 1. 联合、边缘、条件和 Bayes

条件概率定义：

$$
p(z\mid x)=\frac{p(x,z)}{p(x)}.
$$

因此：

$$
p(x,z)=p(x)p(z\mid x).
$$

同理：

$$
p(x,z)=p(z)p(x\mid z).
$$

两者相等：

$$
p(x)p(z\mid x)=p(z)p(x\mid z),
$$

从而：

$$
\boxed{
p(z\mid x)=\frac{p(z)p(x\mid z)}{p(x)}
}.
$$

边缘化：

$$
p(x)=\int p(x,z)dz.
$$

带入生成分解：

$$
\boxed{
p_\theta(x)=\int p(z)p_\theta(x\mid z)dz
}.
$$

---

## 2. ELBO 推导一：从 KL gap 出发

KL 定义：

$$
D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p_\theta(z\mid x))
=
\mathbb E_{q_\phi(z\mid x)}
\left[
\log\frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}
\right].
$$

Bayes：

$$
p_\theta(z\mid x)
=
\frac{p_\theta(x,z)}{p_\theta(x)}.
$$

代入：

$$
\begin{aligned}
D_{\mathrm{KL}}(q_\phi\Vert p_\theta(z\mid x))
&=
\mathbb E_q
\left[
\log q_\phi(z\mid x)
-
\log p_\theta(x,z)
+
\log p_\theta(x)
\right]\\
&=
\log p_\theta(x)
+
\mathbb E_q
\left[
\log q_\phi(z\mid x)
-
\log p_\theta(x,z)
\right].
\end{aligned}
$$

因为 $x$ 已固定，$\log p_\theta(x)$ 与 $z$ 无关，可以直接移出期望。

移项：

$$
\log p_\theta(x)
=
\mathbb E_q
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right]
+
D_{\mathrm{KL}}(q_\phi\Vert p_\theta(z\mid x)).
$$

定义：

$$
\boxed{
\operatorname{ELBO}(x)
=
\mathbb E_q
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right]
}
$$

于是：

$$
\boxed{
\log p_\theta(x)
=
\operatorname{ELBO}(x)
+
D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p_\theta(z\mid x))
}
$$

由 KL 非负：

$$
\boxed{
\operatorname{ELBO}(x)\le \log p_\theta(x)
}.
$$

当且仅当：

$$
q_\phi(z\mid x)=p_\theta(z\mid x)
$$

几乎处处成立时，KL 为 0，下界贴紧真正 log-likelihood。

---

## 3. ELBO 推导二：Jensen + importance weighting 视角

从边缘似然开始：

$$
p_\theta(x)=\int p_\theta(x,z)dz.
$$

乘除同一个 $q_\phi(z\mid x)$：

$$
p_\theta(x)
=
\int q_\phi(z\mid x)
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}dz.
$$

因此：

$$
\boxed{
p_\theta(x)
=
\mathbb E_{q_\phi(z\mid x)}
\left[
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right]
}
$$

这解释了为什么 ELBO 中会自然出现：

$$
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}.
$$

它不是拍脑袋凑出来的，而是把难积分改写成“从一个容易采样的 $q$ 中取样，再用比值修正”的形式。

取 log：

$$
\log p_\theta(x)
=
\log
\mathbb E_q
\left[
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
$$

因为 $\log$ 是凹函数，Jensen 不等式给出：

$$
\log \mathbb E[Y]
\ge
\mathbb E[\log Y].
$$

令：

$$
Y=\frac{p_\theta(x,z)}{q_\phi(z\mid x)},
$$

得到：

$$
\boxed{
\log p_\theta(x)
\ge
\mathbb E_q
\left[
\log
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right]
}
$$

右边正是 ELBO：

$$
\boxed{
\operatorname{ELBO}(x)
=
\mathbb E_q
\left[
\log
\frac{p(z)p_\theta(x\mid z)}{q_\phi(z\mid x)}
\right]
}.
$$

---

## 4. 把 ELBO 拆成 reconstruction 与 prior KL

从：

$$
\operatorname{ELBO}(x)
=
\mathbb E_q
[
\log p(z)+\log p_\theta(x\mid z)-\log q_\phi(z\mid x)
]
$$

开始。

重排：

$$
\begin{aligned}
\operatorname{ELBO}(x)
&=
\mathbb E_q[\log p_\theta(x\mid z)]
+
\mathbb E_q[\log p(z)-\log q_\phi(z\mid x)]\\
&=
\mathbb E_q[\log p_\theta(x\mid z)]
-
\mathbb E_q
\left[
\log\frac{q_\phi(z\mid x)}{p(z)}
\right]\\
&=
\boxed{
\mathbb E_q[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))
}.
\end{aligned}
$$

所以负 ELBO：

$$
\boxed{
-\operatorname{ELBO}(x)
=
\underbrace{-\mathbb E_q[\log p_\theta(x\mid z)]}_{\text{negative log-likelihood / reconstruction}}
+
\underbrace{D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))}_{\text{prior KL}}
}.
$$

---

## 5. 两个 KL 为什么完全不是同一件事？

### 5.1 inference gap

$$
D_{\mathrm{KL}}(
q_\phi(z\mid x)
\Vert
p_\theta(z\mid x)
)
$$

出现在恒等式：

$$
\log p_\theta(x)-\operatorname{ELBO}(x)
=
D_{\mathrm{KL}}(q_\phi\Vert p_\theta(z\mid x)).
$$

它测量 encoder 的近似后验离真正后验有多远。

### 5.2 prior regularization

$$
D_{\mathrm{KL}}(
q_\phi(z\mid x)
\Vert
p(z)
)
$$

出现在可计算的 ELBO 训练式中。它约束每个样本的 posterior 不要任意漂离统一 prior。

一个是“下界 gap”，一个是“训练式的 latent regularizer”。

---

## 6. 标准高斯 prior 下的 KL 闭式解

设一维：

$$
q(z)=\mathcal N(\mu,\sigma^2),
\qquad
p(z)=\mathcal N(0,1).
$$

高斯密度：

$$
\log q(z)
=
-\frac12\log(2\pi\sigma^2)
-\frac{(z-\mu)^2}{2\sigma^2},
$$

$$
\log p(z)
=
-\frac12\log(2\pi)
-\frac{z^2}{2}.
$$

所以：

$$
D_{\mathrm{KL}}(q\Vert p)
=
\mathbb E_q[\log q(z)-\log p(z)].
$$

代入：

$$
\begin{aligned}
D_{\mathrm{KL}}
&=
\frac12
\mathbb E_q
\left[
-\log\sigma^2
-\frac{(z-\mu)^2}{\sigma^2}
+z^2
\right].
\end{aligned}
$$

利用：

$$
\mathbb E_q[(z-\mu)^2]=\sigma^2,
$$

以及：

$$
\mathbb E_q[z^2]=\mu^2+\sigma^2,
$$

得到：

$$
\boxed{
D_{\mathrm{KL}}(q\Vert p)
=
\frac12
\left(
\mu^2+\sigma^2-1-\log\sigma^2
\right)
}.
$$

对 $d$ 维对角高斯，各维相加：

$$
\boxed{
D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))
=
\frac12\sum_{j=1}^d
\left(
\mu_j^2+\sigma_j^2-1-\log\sigma_j^2
\right)
}
$$

等价写法：

$$
\boxed{
D_{\mathrm{KL}}
=
-\frac12\sum_j
\left(
1+\log\sigma_j^2-
\mu_j^2-
\sigma_j^2
\right)
}.
$$

如果代码存的是 `logvar = log(sigma^2)`，则：

$$
\sigma^2=e^{\mathrm{logvar}},
$$

因此 PyTorch 常写：

```python
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
```

---

## 7. 为什么 Bernoulli decoder 对应 BCE？

对于像素 $x_i\in\{0,1\}$，设 decoder 输出 Bernoulli 参数：

$$
p_\theta(x\mid z)
=
\prod_i
\pi_i(z)^{x_i}
(1-\pi_i(z))^{1-x_i}.
$$

取 log：

$$
\log p_\theta(x\mid z)
=
\sum_i
\left[
x_i\log\pi_i
+(1-x_i)\log(1-\pi_i)
\right].
$$

负号后：

$$
-\log p_\theta(x\mid z)
=
\operatorname{BCE}(x,\pi).
$$

实践中 decoder 输出 logits $a_i$，令：

$$
\pi_i=\sigma(a_i),
$$

使用数值稳定的 `binary_cross_entropy_with_logits`。

---

## 8. 为什么 Gaussian decoder 常对应 MSE？

如果假设：

$$
p_\theta(x\mid z)
=
\mathcal N(f_\theta(z),\sigma_x^2 I),
$$

则：

$$
\log p_\theta(x\mid z)
=
C
-
\frac{1}{2\sigma_x^2}
\|x-f_\theta(z)\|_2^2.
$$

当 $\sigma_x^2$ 固定时，最大化 log-likelihood 等价于最小化 MSE（差一个常数和缩放系数）。

所以“reconstruction loss”并不是天生就是 BCE 或 MSE；它取决于我们选择了怎样的 observation likelihood $p_\theta(x\mid z)$。

---

## 9. Reparameterization trick 的梯度逻辑

我们要优化：

$$
\mathbb E_{q_\phi(z\mid x)}[f_\theta(z)].
$$

但 $z$ 的采样分布依赖 $\phi$。对于高斯，可以令：

$$
\epsilon\sim\mathcal N(0,I),
$$

并定义：

$$
\boxed{
z=g_\phi(x,\epsilon)
=
\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon
}.
$$

于是期望变成：

$$
\mathbb E_{\epsilon\sim\mathcal N(0,I)}
[f_\theta(g_\phi(x,\epsilon))].
$$

现在采样分布不再依赖 $\phi$；$\phi$ 只出现在确定性函数 $g_\phi$ 中，因此可以用普通 backprop：

$$
\nabla_\phi
\mathbb E_\epsilon[f(g_\phi(\epsilon))]
=
\mathbb E_\epsilon
[
\nabla_z f(z)\nabla_\phi g_\phi(\epsilon)
].
$$

这叫 pathwise gradient estimator。

---

## 10. 单样本 Monte Carlo ELBO estimator

理论上：

$$
\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
$$

仍是一个期望。

训练时常取一个：

$$
\epsilon\sim\mathcal N(0,I),
$$

$$
z=\mu+\sigma\odot\epsilon,
$$

然后用：

$$
\log p_\theta(x\mid z)
$$

作为单样本 Monte Carlo 估计。

对 minibatch $B$：

$$
\widehat{\mathcal L}
=
\frac{N}{|B|}
\sum_{x_i\in B}
\operatorname{ELBO}(x_i)
$$

可作为全数据集目标的无偏 minibatch 缩放估计；实际优化器也常直接使用 batch mean，而把全局常数吸收到学习率中。

---

## 11. $q(z)$：aggregated posterior 的推导

定义 inference joint：

$$
q(x,z)=p_{\mathrm{data}}(x)q_\phi(z\mid x).
$$

边缘化 $x$：

$$
\boxed{
q(z)
=
\int p_{\mathrm{data}}(x)q_\phi(z\mid x)dx
}.
$$

离散数据集的经验分布下：

$$
\boxed{
q(z)
\approx
\frac1N
\sum_{i=1}^N
q_\phi(z\mid x_i)
}.
$$

因此 $q(z)$ 是所有 per-example posterior 的混合。

注意：标准 VAE 的 ELBO 中逐样本惩罚的是：

$$
D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)),
$$

而不是直接写：

$$
D_{\mathrm{KL}}(q(z)\Vert p(z)).
$$

二者不是同一个约束。

---

## 12. KL 非对称与 mode behavior 的数学来源

### 12.1 $D_{\mathrm{KL}}(P\Vert Q)$

$$
D_{\mathrm{KL}}(P\Vert Q)
=
\mathbb E_{x\sim P}
\left[
\log\frac{P(x)}{Q(x)}
\right].
$$

平均权重来自 $P$。若某区域 $P(x)>0$ 但 $Q(x)\to0$：

$$
\log\frac{P(x)}{Q(x)}\to+\infty.
$$

所以遗漏 $P$ 的支持区域会很贵。

### 12.2 $D_{\mathrm{KL}}(Q\Vert P)$

$$
D_{\mathrm{KL}}(Q\Vert P)
=
\mathbb E_{x\sim Q}
\left[
\log\frac{Q(x)}{P(x)}
\right].
$$

平均权重来自 $Q$。$Q$ 几乎不去的地方几乎不参与平均；但如果 $Q$ 把质量放到 $P(x)\approx0$ 的位置，则代价很大。

在受限单峰 $Q$ 去拟合多峰 $P$ 时，这常导致选一个峰而不是横跨低密度区，因此被称为 mode-seeking。

这是一种常见近似行为，不是脱离具体参数化后的普遍定理。

---

## 13. 从 ELBO 到代码的精确对应

对于一条样本 $x$：

1. encoder：

$$
(\mu,\log\sigma^2)=\operatorname{Enc}_\phi(x)
$$

2. reparameterize：

$$
\epsilon\sim\mathcal N(0,I),
\qquad
z=\mu+\exp(\tfrac12\log\sigma^2)\odot\epsilon
$$

3. decoder：

$$
\text{logits}=\operatorname{Dec}_\theta(z)
$$

4. reconstruction NLL：

$$
\mathcal L_{\mathrm{rec}}
=-\log p_\theta(x\mid z)
$$

5. KL：

$$
\mathcal L_{\mathrm{KL}}
=D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert\mathcal N(0,I))
$$

6. 总 loss：

$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{rec}}
+
\mathcal L_{\mathrm{KL}}
=
-\widehat{\operatorname{ELBO}}
}
$$

7. backprop 同时更新：

$$
\theta\quad\text{和}\quad\phi.
$$

---

## 14. 一个最后的结构化结论

VAE 的数学不是“先有一个 reconstruction loss，再人为加一个 KL regularizer”这么简单。

更本质的顺序是：

$$
\boxed{
\text{最大化 }\log p_\theta(x)
}
$$

但边缘似然包含难积分，于是引入：

$$
q_\phi(z\mid x)
$$

构造：

$$
\operatorname{ELBO}
$$

而把 ELBO 代数展开后，**自然得到**：

$$
\boxed{
\text{reconstruction term}
-
\text{prior KL term}
}
$$

所以这两个项首先是**同一个概率目标的分解结果**，然后我们才可以再从表示学习角度解释它们的行为。
