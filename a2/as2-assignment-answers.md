# CS 224n Assignment #2: `word2vec` (44 Points)

## 1 Written: Understanding `word2vec` (26 points)

(a) (3 points) Show that the naive-softmax loss given in Equation (2) is the same as the cross-entropy loss between $y$ and $\hat{y}$; i.e., show that

$$- \sum_{w\isin{\text{Vocab}}} y_{w} \log{\hat{y}_{w}} = -\log{\hat{y}_{o}}$$

Your answer should be one line.

### Answer:

The true empirical distribution (i.e., the ground truth) $y$ is a one-hot vector where $y_{w} = 1$ when $w = o$ and $y_{w} = 0$ when $w \neq o$. Mathematically,

$$
y_{w} = \begin{cases} 1 \text{ if } w = o & \\ 0 \text{ if } w \neq o\end{cases}
$$

As such, considering the cross-entropy loss $J_{\text{cross-entropy}}$,

$$
J_{\text{cross-entropy}}(y, \hat{y}) = -\sum_{w\isin{\text{Vocab}}} y_{w} \log{(\hat{y}_{w})} \\[0.2in]
= -[y_{1}\log{(\hat{y}_{1})} + ... + y_{o}\log{(\hat{y}_{o})} + ... + y_{w}\log{(\hat{y}_{w})}] \\[0.2in]
= -y_{o}\log{(\hat{y}_{o}}) \\[0.2in]
= -\log{(\hat{y}_{o})} \\[0.2in]
= -\log{P(O=o|C=c)} \\[0.2in]
= \boxed{J_{\text{naive-softmax}}(v_{c}, o, U)}
$$

(b) (5 points) Compute the partial derivative of $J_{\text{naive-softmax}}(v_{c}, o, U)$ with respect to $v_{c}$. Please write your answer in terms of $y$, $\hat{y}$, and $U$. Note that in this course, we expect your final answers to follow the shape convention. This means that the partial derivative of any function $f(x)$ with respect to $x$ should have the same shape as $x$. For this subpart, please present your answer in vectorized form. In particular, you may not refer to specific elements of $y$, $\hat{y}$, and $U$ in your final answer (such as $y_{1}$, $y_{2}$, ...).

### Answer:

$$
\frac{\delta}{\delta{v_{c}}} J_{\text{naive-softmax}}(v_{c}, o, U) = -\frac{\delta}{\delta{v_{c}}} \log{P(O=o|C=c)} \\[0.2in]
= - \frac{\delta}{\delta{v_{c}}} \log{\frac{\exp{(u^{T}_{o}v_{c})}}{\sum_{w\isin{\text{Vocab}}}\exp{(u^{T}_{w}v_{c})}}} \\[0.2in]
= -\frac{\delta}{\delta{v_{c}}} \log{\exp{(u_{o}^{T}v_{c})}} + \frac{\delta}{\delta{v_{c}}} \log{\sum_{w\isin{\text{Vocab}}} \exp{(u_{w}^T}v_{c})} \\[0.2in]
= -\frac{\delta}{\delta{v_{c}}} u_{o}^{T}v_{c} + \frac{1}{\sum_{w\isin{\text{Vocab}}} \exp{(u_{w}^T}v_{c})} \frac{\delta}{\delta{v_{c}}} \sum_{w\isin{\text{Vocab}}} \exp{(u_{w}^{T}v_{c})} \\[0.2in]
= -u_{o} + \frac{1}{\sum_{w\isin{\text{Vocab}}} \exp{(u_{w}^T}v_{c})} \sum_{w\isin{\text{Vocab}}} \frac{\delta}{\delta{v_{c}}} \exp{(u_{w}^{T}v_{c})} \\[0.2in]
= -u_{o} + \frac{1}{\sum_{w\isin{\text{Vocab}}} \exp{(u_{w}^T}v_{c})} \sum_{w\isin{\text{Vocab}}} \exp{(u_{w}^{T}v_{c})} \frac{\delta}{\delta{v_{c}}} u_{w}^{T}v_{c} \\[0.2in]
= -u_{o} + \frac{\sum_{w\isin{\text{Vocab}}} \exp{(u_{w}^{T}v_{c})} u_{w}}{\sum_{w\isin{\text{Vocab}}} \exp{(u_{w}^{T}v_{c})}} \\[0.2in]
$$

Since $y_{w} = \begin{cases} 1 \text{ if } w = o \\ 0 \text{ if } w \neq o\end{cases}$, we can write $u_{o}$ as $\sum_{w\isin{\text{Vocab}}} y_{w}u_{w}$. As such,

$$
= -\sum_{w\isin{\text{Vocab}}} y_{w}u_{w} + \sum_{w\isin{\text{Vocab}}} \frac{\exp{(u_{w}^{T})v_{c}}}{\sum_{k\isin{\text{Vocab}}} \exp{(u_{k}^{T}v_{c})}}u_{w}
$$

Since $\frac{\exp{(u_{w}^{T})v_{c}}}{\sum_{k\isin{\text{Vocab}}} \exp{(u_{k}^{T}v_{c})}}$ is the conditional probability distribution, $\hat{y}_{w}$,

$$
= -\sum_{w\isin{\text{Vocab}}} y_{w}u_{w} + \sum_{w\isin{\text{Vocab}}} \hat{y}_{w}u_{w} \\[0.2in]
= \sum_{w\isin{\text{Vocab}}} (-y_{w}u_{w} + \hat{y}_{w}u_{w}) \\[0.2in]
= \sum_{w\isin{\text{Vocab}}} u_{w}(-y_{w} + \hat{y}_{w}) \\[0.2in]
= \sum_{w\isin{\text{Vocab}}} u_{w}(\hat{y}_{w} -y_{w}) \\[0.2in]
$$

Note that $y$ is a one-hot vector with $1$ at word $o$ and $0$ at all other positions. Vectorizing the above equation in terms of $y$, $\hat{y}$ and $U$ gives us,

$$
\boxed{\frac{\delta}{\delta{v_{c}}} J_{\text{naive-softmax}}(v_{c}, o, U) = U(\hat{y} - y)}
$$

(c) (5 points) Compute the partial derivatives of $J_{\text{naive-softmax}}(v_{c}, o, U)$ with respect to each of the ‘outside’ word vectors, $u_{w}$’s. There will be two cases: when $w = o$, the true ‘outside’ word vector, and $w \neq o$, for all other words. Please write your answer in terms of $y$, $\hat{y}$, and $v_{c}$. In this subpart, you may use specific elements within these terms as well, such as ($y_{1}$, $y_{2}$, ...).

### Answer:

$$
\frac{\delta}{\delta{u_{w}}} J_{\text{naive-softmax}}(v_{c}, o, U) = -\frac{\delta}{\delta{u_{w}}} \log{P(O=o|C=c)} \\[0.2in]
= - \frac{\delta}{\delta{u_{w}}} \log{\frac{\exp{u^{T}_{o}v_{c}}}{\sum_{w\isin{\text{Vocab}}}\exp{(u^{T}_{w}v_{c})}}} \\[0.2in]
= -\frac{\delta}{\delta{u_{w}}} \log{\exp{(u_{o}^{T}v_{c})}} + \frac{\delta}{\delta{u_{w}}} \log{\sum_{w\isin{\text{Vocab}}} \exp{(u_{w}^T}v_{c})}
$$

$\text{When } w \neq o:$

$$
\frac{\delta}{\delta{u_{w}}} J_{\text{naive-softmax}}(v_{c}, o, U) \\[0.2in]
= \frac{\delta}{\delta{u_{w}}} \log{\sum_{w\isin{\text{Vocab}}} \exp{(u_{w}^T}v_{c})} \\[0.2in]
= \frac{1}{\sum_{k\isin{\text{Vocab}}} \exp{(u_{k}^T}v_{c})} \frac{\delta}{\delta{u_{w}}} \sum_{\substack{x\isin{\text{Vocab}} \\ x \neq o}} \exp{(u_{w}^{T}v_{c})} \\[0.2in]
= \frac{1}{\sum_{k\isin{\text{Vocab}}} \exp{(u_{k}^T}v_{c})} \frac{\delta}{\delta{u_{w}}} \exp{(u_{x}^{T}v_{c}}) \\[0.2in]
= \frac{\exp{(u_{w}^{T}v_{c})}}{\sum_{k\isin{\text{Vocab}}} \exp{(u_{k}^T}v_{c})} v_{c}
$$

Since $\frac{\exp{(u_{w}^{T}v_{c})}}{\sum_{k\isin{\text{Vocab}}} \exp{(u_{k}^{T}v_{c})}}$ is the conditional probability distribution, $\hat{y}_{w}$,

$$
\boxed{\frac{\delta}{\delta{u_{w}}} J_{\text{naive-softmax}}(v_{c}, o, U) = \hat{y}_{w}v_{c}}
$$

$\text{When } w = o:$

$$
\frac{\delta}{\delta{u_{o}}} J_{\text{naive-softmax}}(v_{c}, o, U) \\[0.2in]
= -\frac{\delta}{\delta{u_{o}}} \log{\exp{(u_{o}^{T}v_{c})}} + \frac{\delta}{\delta{u_{o}}} \log{\sum_{w\isin{\text{Vocab}}} \exp{(u_{w}^T}v_{c})} \\[0.2in]
= -v_{c} + \frac{\exp{(u_{o}^{T}v_{c})}}{\sum_{k\isin{\text{Vocab}}} \exp{(u_{k}^T}v_{c})} v_{c} \\[0.2in]
= ( \frac{\exp{(u_{o}^{T}v_{c})}}{\sum_{k\isin{\text{Vocab}}} \exp{(u_{k}^T}v_{c})} - 1) v_{c}
$$

Since $\frac{\exp{(u_{o}^{T}v_{c})}}{\sum_{k\isin{\text{Vocab}}} \exp{(u_{k}^{T}v_{c})}}$ is the conditional probability distribution, $\hat{y}_{o}$,

$$
\boxed{\frac{\delta}{\delta{u_{o}}} J_{\text{naive-softmax}}(v_{c}, o, U) = (\hat{y}_{o} - 1)v_{c}}
$$

Consolidating both cases,

$$
\frac{\delta{J}}{\delta{u_{w}}} = \begin{cases} (\hat{y}_{o} - 1)v_{c} \text{ if } w = o \\ \hat{y}_{w}v_{c} \text{ otherwise }\end{cases}
$$

(d) (1 point) Compute the partial derivative of $J_{\text{naive-softmax}}(v_{c}, o, U)$ with respect to $U$. Please write your answer in terms of $\frac{\delta{J(v_{c}, o, U)}}{\delta{u_{1}}}, \frac{\delta{J(v_{c}, o, U)}}{\delta{u_{2}}}, ..., \frac{\delta{J(v_{c}, o, U)}}{\delta{u_{|Vocab|}}}$. The solution should be one or two lines long.

### Answer:

The derivative of a scalar $y$ by a matrix $A$ is given by,

$$
\frac{\delta{y}}{\delta{A_{m \times n}}} =
\begin{bmatrix}
\frac{\delta{y}}{\delta{A_{11}}} & \frac{\delta{y}}{\delta{A_{12}}} &\cdots & \frac{\delta{y}}{\delta{A_{1n}}} \\
\vdots & \vdots & \ddots & \vdots\\
\frac{\delta{y}}{\delta{A_{m1}}} & \frac{\delta{y}}{\delta{A_{m2}}} & \cdots & \frac{\delta{y}}{\delta{A_{mn}}}
\end{bmatrix}
$$

Given $u_{w}$ represents the vector for "outside" word $w$, the derivative of $J_{\text{naive-softmax}}$ (which is a scalar) by $U$ (which is a matrix) is,

$$
\frac{\delta{J_{\text{naive-softmax}}(v_{c}, o, U)}}{\delta{U}} =
\begin{bmatrix}
\frac{\delta{J}}{\delta{u_{1}}} & \frac{\delta{J}}{\delta{u_{2}}} & \cdots & \frac{\delta{J}}{\delta{u_{\text{Vocab}}}}
\end{bmatrix}
$$

where,

$$
\frac{\delta{J}}{\delta{u_{w}}} = \begin{cases} (\hat{y} - 1)v_{c} \text{ if } w = o \\ \hat{y}v_{c} \text{ if } w \neq o \end{cases}
$$

(e) (3 Points) The sigmoid function is given by Equation 4:

$$
\sigma{(x)} = \frac{1}{1 + e^{-x}} = \frac{e^{x}}{e^{x} + 1}
$$

Please compute the derivative of $\sigma{(x)}$ with respect to $x$, where $x$ is a scalar. Hint: you may want to write your answer in terms of $\sigma{(x)}$.

### Answer:

$$
\frac{\delta{\sigma}}{\delta{x}} = \frac{\delta}{\delta{x}} \left[\frac{e^{x}}{e^{x} + 1}\right]
$$

Using the quotient rule,

$$
= \frac{e^{x}(e^{x} + 1) - e^{2x}}{(e^{x} + 1)^{2}} \\[0.2in]
= \frac{e^{x}}{(e^{x} + 1)^{2}} \\[0.2in]
= \frac{e^{x}}{e^{x} + 1} \frac{1}{e^{x} + 1} \\[0.2in]
= \sigma(x) \cdot \frac{1}{e^{x} + 1}
$$

We can multiply with $\frac{e^{-x}}{e^{-x}}$ which is essentially equivalent to $1$,

$$
= \sigma(x) \cdot \left[ \frac{1}{e^{x} + 1} \right] \times \frac{e^{-x}}{e^{-x}} \\[0.2in]
= \sigma{(x)} \cdot \frac{e^{-x}}{1 + e^{-x}} \\[0.2in]
= \sigma{(x)} \cdot \left[ \frac{1+e^{-x}}{1+e^{-x}} - \frac{1}{1+e^{-x}} \right] \\[0.2in]
= \boxed{\sigma{(x)}(1 - \sigma{(x)})}
$$

(f) (4 points) Now we shall consider the Negative Sampling loss, which is an alternative to the Naive Softmax loss. Assume that $K$ negative samples (words) are drawn from the vocabulary. For simplicity of notation we shall refer to them as $w_{1}, w_{2}, ..., w_{K}$ and their outside vectors as $u_{1}, ..., u_{K}$. For this question, assume that the $K$ negative samples are distinct. In other words, $i \neq j$ implies $w_{i} \neq w_{j}$ for $i, j \isin \{1, ..., K\}$. Note that $o \notin \{w_{1}, ..., w_{K}\}$. For a center word $c$ and an outside word $o$, the negative sampling loss function is given by:

$$
J_{\text{neg-sample}}(v_{c}, o, U) = -\log{(\sigma{(u_{o}^{T}v_{c})})} - \sum_{k=1}^{K} \log{(\sigma{(-u_{k}^{T}v_{c})})}
$$

for a sample $w_{1}, ..., w_{K}$, where $\sigma{(\cdot)}$ is the sigmoid function.

Please repeat parts (b) and (c), computing the partial derivatives of $J_{\text{neg-sample}}$ with respect to $v_{c}$, with respect to $u_{o}$, and with respect to a negative sample $u_{k}$. Please write your answers in terms of the vectors $u_{o}, v_{c}$ and $u_{k}$, where $k \isin \left[1, K\right]$. After you’ve done this, describe with one sentence why this loss function is much more efficient to compute than the naive-softmax loss. Note, you should be able to use your solution to part (e) to help compute the necessary gradients here.

### Answer:

The loss function contains two terms:

(i) The first term is the log of the probability that the center word and true outside word came from the corpus.

(ii) The second term is the sum of the logs of the probabilities that the center word and outside context words did not come from the corpus.

1. Computing the partial derivatives of $J_{\text{neg-sample}}$ w.r.t $v_{c}$ which is the word vector of the center word, $c$.

$$
\frac{\delta{J_{\text{neg-sample}}(v_{c}, o, U)}}{\delta{v_{c}}} = \frac{\delta}{\delta{v_{c}}} \left[ -\log{(\sigma{(u_{o}^{T}v_{c})})} - \sum_{k=1}^{K} \log{(\sigma{(-u_{k}^{T}v_{c})})} \right] \\[0.2in]
= \frac{\delta}{\delta{v_{c}}} \left[ -log{(\sigma{(u_{o}^{T}v_{c})})} \right] - \frac{\delta}{\delta{v_{c}}} \left[ \sum_{k=1}^{K} \log{(\sigma{(-u_{k}^{T}v_{c})})} \right] \\[0.2in]
= \frac{\delta}{\delta{v_{c}}} \left[ -log{(\sigma{(u_{o}^{T}v_{c})})} \right] - \sum_{k=1}^{K} \left[ \frac{\delta}{\delta{v_{c}}} \log{(\sigma{(-u_{k}^{T}v_{c})})} \right] \\[0.2in]
= \left(-\frac{1}{\sigma{(u_{o}^{T}v_{c})}} \right) \sigma{(u_{o}^{T}v_{c})}(1 - \sigma{(u_{o}^{T}v_{c})}) u_{o} - \sum_{k=1}^{K} \frac{u_{k}\sigma{(-u_{k}^{T}v_{c})} (1 - \sigma{(-u_{k}^{T}v_{c})})}{\sigma{(-u_{k}^{T}v_{c})}} \\[0.2in]
= -u_{o}(1 - \sigma{(u_{o}^{T}v_{c})}) - \sum_{k=1}^{K} u_{k}(1 - \sigma{(-u_{k}^{T}v_{c})}) \\[0.2in]
= \boxed{u_{o}(\sigma{(u_{o}^{T}v_{c})} - 1) - \sum_{k=1}^{K} u_{k}(\sigma{(-u_{k}^{T}v_{c})} - 1)}
$$

2. Computing the partial derivatives of $J_{\text{neg-sample}}$ w.r.t $u_{o}$ which is the word vector of the outside word, $o$.

$$
\frac{\delta{J_{\text{neg-sample}}(v_{c}, o, U)}}{\delta{u_{o}}} = \frac{\delta}{\delta{u_{o}}} \left[ -\log{(\sigma{(u_{o}^{T}v_{c})})} - \sum_{k=1}^{K} \log{(\sigma{(-u_{k}^{T}v_{c})})} \right] \\[0.2in]
= \frac{\delta}{\delta{u_{o}}} \left[ -log{(\sigma{(u_{o}^{T}v_{c})})} \right] - \frac{\delta}{\delta{u_{o}}} \left[ \sum_{k=1}^{K} \log{(\sigma{(-u_{k}^{T}v_{c})})} \right]
$$

Since $o \notin \{w_{1}, ..., w_{K}\}$, $\frac{\delta}{\delta{u_{o}}} \left[ \sum_{k=1}^{K} \log{(\sigma{(-u_{k}^{T}v_{c})})} \right] = 0$. As such,

$$
\frac{\delta{J_{\text{neg-sample}}}}{\delta{u_{o}}} = \frac{\delta}{\delta{u_{o}}} \left[ -log{(\sigma{(u_{o}^{T}v_{c})})} \right] - 0
$$

Using part (e), $\frac{\delta{\sigma}}{\delta{x}} = \sigma{(x)}(1 - \sigma{(x)})$. Hence,

$$
\frac{\delta{J_{\text{neg-sample}}(v_{c}, o, U)}}{\delta{u_{o}}} = \left(-\frac{1}{\sigma{(u_{o}^{T}v_{c})}} \right) \sigma{(u_{o}^{T}v_{c})}(1 - \sigma{(u_{o}^{T}v_{c})}) v_{c} \\[0.2in]
= -v_{c}(1 - \sigma{(u_{o}^{T}v_{c})}) \\[0.2in]
= \boxed{= v_{c}(\sigma{(u_{o}^{T}v_{c})} - 1)}
$$

3. Computing the partial derivatives of $J_{\text{neg-sample}}$ w.r.t $u_{k}$ which is the word vector for one of the $K$ negative samples.

$$
\frac{\delta{J_{\text{neg-sample}}(v_{c}, o, U)}}{\delta{u_{k}}} = \frac{\delta}{\delta{u_{k}}} \left[ -\log{(\sigma{(u_{o}^{T}v_{c})})} - \sum_{k=1}^{K} \log{(\sigma{(-u_{k}^{T}v_{c})})} \right] \\[0.2in]
= \frac{\delta}{\delta{u_{k}}} \left[ -log{(\sigma{(u_{o}^{T}v_{c})})} \right] - \frac{\delta}{\delta{u_{k}}} \left[ \sum_{k=1}^{K} \log{(\sigma{(-u_{k}^{T}v_{c})})} \right]
$$

Since $o \notin \{w_{1}, ..., w_{K}\}$, $\frac{\delta}{\delta{u_{k}}} \left[ -log{(\sigma{(u_{o}^{T}v_{c})})} \right] = 0$. As such,

$$
\frac{\delta{J_{\text{neg-sample}}(v_{c}, o, U)}}{\delta{u_{k}}} = 0 - \frac{\delta}{\delta{u_{k}}} \left[ \sum_{k=1}^{K} \log{(\sigma{(-u_{k}^{T}v_{c})})} \right]
$$

Since the derivative of a sum is the sum of derivatives,

$$
\frac{\delta{J_{\text{neg-sample}}(v_{c}, o, U)}}{\delta{u_{k}}} = -\sum_{k=1}^{K} \left[ \frac{\delta}{\delta{u_{k}}} \log{(\sigma{(-u_{k}^{T}v_{c})})} \right]
$$

Now, derivatives of all the terms with $u_{w}$ where $w \neq k$ are $0$ while the derivative of the term with $u_{w}$ where $w = k$ remains.

Using part (e), $\frac{\delta{\sigma}}{\delta{x}} = \sigma{(x)}(1 - \sigma{(x)})$. Hence,

$$
\frac{\delta{J_{\text{neg-sample}}(v_{c}, o, U)}}{\delta{u_{k}}} = -\frac{v_{c}\sigma{(-u_{k}^{T}v_{c})} (1 - \sigma{(-u_{k}^{T}v_{c})})}{\sigma{(-u_{k}^{T}v_{c})}} \\[0.2in]
= \boxed{v_{c}(\sigma{(-u_{k}^{T}v_{c}) - 1}), \forall k \isin [1, K]}
$$

By using the negative sampling loss, we only need to go through $O(K)$ samples, while the naive-softmax loss requires traversing through the whole vocabulary, which is is typically much greater than $K$.

(g) (2 point) Now we will repeat the previous exercise, but without the assumption that the $K$ sampled words are distinct. Assume that $K$ negative samples (words) are drawn from the vocabulary. For simplicity of notation we shall refer to them as $w_{1}, w_{2}, ..., w_{K}$ and their outside vectors as $u_{1}, ..., u_{K}$. In this question, you may not assume that the words are distinct. In other words, $w_{i} = w_{j}$ may be true when $i \neq j$ is true. Note that $o \notin \{w_{1}, ..., w_{K}\}$. For a center word $c$ and an outside word $o$, the negative sampling loss function is given by:

$$
J_{\text{neg-sample}}(v_{c}, o, U) = -\log{(\sigma{(u_{o}^{T}v_{c})})} - \sum_{k=1}^{K} \log{(\sigma{(-u_{k}^{T}v_{c})})}
$$

for a sample $w_{1}, ..., w_{K}$, where $\sigma{(\cdot)}$ is the sigmoid function.

Compute the partial derivative of $J_{\text{neg-sample}}$ with respect to a negative sample $u_{k}$. Please write your answers in terms of the vectors $v_{c}$ and $u_{k}$, where $k \isin [1, K]$. Hint: break up the sum in the loss function into two sums: a sum over all sampled words equal to $u_{k}$ and a sum over all sampled words not equal to $u_{k}$.

### Answer:

$$
\frac{\delta{J_{\text{neg-sample}}(v_{c}, o, U)}}{\delta{u_{k}}} = \frac{\delta}{\delta{u_{k}}} \left[ -\log{(\sigma{(u_{o}^{T}v_{c})})} - \sum_{k=1}^{K} \log{(\sigma{(-u_{k}^{T}v_{c})})} \right] \\[0.2in]
= \frac{\delta}{\delta{u_{k}}} \left[ -log{(\sigma{(u_{o}^{T}v_{c})})} \right] - \frac{\delta}{\delta{u_{k}}} \left[ \sum_{k=1}^{K} \log{(\sigma{(-u_{k}^{T}v_{c})})} \right]
$$

Since $o \notin \{w_{1}, ..., w_{K}\}$, $\frac{\delta}{\delta{u_{k}}} \left[ -log{(\sigma{(u_{o}^{T}v_{c})})} \right] = 0$. As such,

$$
\frac{\delta{J_{\text{neg-sample}}(v_{c}, o, U)}}{\delta{u_{k}}} = 0 - \frac{\delta}{\delta{u_{k}}} \left[ \sum_{k=1}^{K} \log{(\sigma{(-u_{k}^{T}v_{c})})} \right]
$$

Note that $K$ negative samples, $i \isin [1, K]$, were drawn from the vocabulary which cannot be assumed to
be distinct. As such, let's break up the sum in the loss function into two sums:

1. sum over all sampled words $w_{i}$ equal to $w_{k}$ and,
2. sum over all sampled words $w_{i}$ not equal to $w_{k}$

Further, note that here we are iterating over the indices of the words $w$ instead of indices of the vectors $u$.

$$
\frac{\delta{J_{\text{neg-sample}}(v_{c}, o, U)}}{\delta{u_{k}}} = - \frac{\delta}{\delta{u_{k}}} \left[ \sum_{i \isin \{1, ..., K\} : w_{i} = w_{k}}  \log{(\sigma{(-u_{i}^{T}v_{c})})} + \sum_{i \isin \{1, ..., K\} : w_{i} \neq w_{k}}  \log{(\sigma{(-u_{i; w_{i} \neq w_{k}}^{T}v_{c})})} \right] \\[0.2in]
= - \left[ \sum_{i \isin \{1, ..., K\} : w_{i} = w_{k}} \frac{\delta}{\delta{u_{k}}} \log{(\sigma{(-u_{i}^{T}v_{c})})} + \sum_{i \isin \{1, ..., K\} : w_{i} \neq w_{k}} \frac{\delta}{\delta{u_{k}}} \log{(\sigma{(-u_{i; w_{i} \neq w_{k}}^{T}v_{c})})} \right]
$$

Now, $\frac{\delta}{\delta{u_{k}}} \log{(\sigma{(-u_{i; w_{i} \neq w_{k}}^{T}v_{c})})} = 0$. Thus,

$$
\frac{\delta{J_{\text{neg-sample}}(v_{c}, o, U)}}{\delta{u_{k}}} = - \left[ \sum_{i \isin \{1, ..., K\} : w_{i} = w_{k}} \frac{\delta}{\delta{u_{k}}} \log{(\sigma{(-u_{i}^{T}v_{c})})} \right]
$$

Or equivalently,

$$
\frac{\delta{J_{\text{neg-sample}}(v_{c}, o, U)}}{\delta{u_{k}}} = - \left[ \sum_{i \isin \{1, ..., K\} : w_{i} = w_{k}} \frac{\delta}{\delta{u_{k}}} \log{(\sigma{(-u_{k}^{T}v_{c})})} \right]
$$

Using the chain rule on $\log$,

$$
\frac{\delta{J_{\text{neg-sample}}(v_{c}, o, U)}}{\delta{u_{k}}} = - \sum_{i \isin \{1, ..., K\} : w_{i} = w_{k}} \frac{-v_{c}\sigma{(-u_{k}^{T}v_{c})}(1 - \sigma{(-u_{k}^{T}v_{c})})}{\sigma{(-u_{k}^{T}v_{c})}} \\[0.2in]
= \sum_{i \isin \{1, ..., K\} : w_{i} = w_{k}} v_{c}(1 - \sigma{(-u_{k}^{T}v_{c})}) \\[0.2in]
= \boxed{\sum_{i \isin \{1, ..., K\} : w_{i} = w_{k}} -v_{c}(\sigma{(-u_{k}^{T}v_{c})} - 1)}
$$

(h) (3 points) Suppose the center word is $c = w_{t}$ and the context window is $[w_{t-m}, ..., w_{t-1}, w_{t}, w_{t+1}, ..., w_{t+m}]$, where $m$ is the context window size. Recall that for the skip-gram version of `word2vec`, the total loss for the context window is:

$$
J_{\text{skip-gram}}(v_{c}, w_{t-m}, ..., w_{t+m}, U) = \sum_{\substack{-m \le j \le m \\ j \ne 0}} J(v_{c}, w_{t+j}, U)
$$

Here, $J(v_{c}, w_{t+j}, U)$ represents an arbitrary loss term for the center word $c = w_{t}$ and outside word $w_{t+j}$. $J(v_{c}, w_{t+j}, U)$ could be $J_{\text{naive-softmax}}(v_{c}, w_{t+j}, U)$ or $J_{\text{neg-sample}}(v_{c}, w_{t+j}, U)$, depending on your implementation.

Write down three partial derivatives:

(i) $\delta{J_{\text{skip-gram}}}(v_{c}, w_{t-m}, ..., w_{t+m}, U)/\delta{U}$

(ii) $\delta{J_{\text{skip-gram}}}(v_{c}, w_{t-m}, ..., w_{t+m}, U)/\delta{v_{c}}$

(iii) $\delta{J_{\text{skip-gram}}}(v_{c}, w_{t-m}, ..., w_{t+m}, U)/\delta{v_{w}} \text{ when } w \ne c$

Write your answers in terms of $\delta{J(v_{c}, w_{t+j}, U)}/\delta{U}$ and $\delta{J(v_{c}, w_{t+j}, U)}/\delta{v_{c}}$. This is very simple –
each solution should be one line.

_Once you’re done: Given that you computed the derivatives of_ $J(v\_{c}, w_{t+j}, U)$ _with respect to all the model parameters_ $U$ _and_ $V$ _in parts (a) to (c), you have now computed the derivatives of the full loss function_ $J_{\text{skip-gram}}$ _with respect to all parameters. You’re ready to implement `word2vec`!_

### Answer:

$$
\frac{\delta{J_{\text{skip-gram}}}(v_{c}, w_{t-m}, ..., w_{t+m}, U)}{\delta{U}} = \sum_{\substack{-m \le j \le m \\ j \ne 0}} \frac{\delta{J(v_{c}, w_{t+j}, U)}}{\delta{U}}

\\[0.2in]

\frac{\delta{J_{\text{skip-gram}}}(v_{c}, w_{t-m}, ..., w_{t+m}, U)}{\delta{v_{c}}} = \sum_{\substack{-m \le j \le m \\ j \ne 0}} \frac{\delta{J(v_{c}, w_{t+j}, U)}}{\delta{v_{c}}}

\\[0.2in]

\frac{\delta{J_{\text{skip-gram}}}(v_{c}, w_{t-m}, ..., w_{t+m}, U)}{\delta{v_{w}}} = 0
$$

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>
