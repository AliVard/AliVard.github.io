---
layout: archive
title: "Learning to Rank"
permalink: /readings/ltr/
author_profile: false
sidebar: toc
redirect_from:
  - /readings/ltr.html
---

### [Unified Off-Policy Learning to Rank: a Reinforcement Learning Perspective](https://arxiv.org/pdf/2306.07528.pdf) (2023)

> In this paper, we unified the ranking process under general stochastic click models as a Markov Decision Process (MDP), and the optimal ranking
could be learned with offline reinforcement learning (RL) directly.

> Our key insight is that the user’s examination and click behavior
summarized by click models has a Markov structure ...
Specifically, the learning to rank problem
now can be viewed as an episodic RL problem [45, 1], where each time step corresponds to a
ranking position, each action selects a document for the position, and the state captures the user’s
examination tendency.
...
We first construct each logged query and ranking data as an episode of reinforcement learning following the MDP formulation. Our dedicated structure for state representation
learning can efficiently capture the dependency information for examination and click generation,
e.g. ranking position in PBM and previous documents in CM and DCM. The algorithm jointly learns
state representation and optimizes the policy, where any off-the-shelf offline RL algorithm can be
applied as a plug-in solver. 

---
### [I3 Retriever: Incorporating Implicit Interaction in Pre-trained Language Models for Passage Retrieval](https://arxiv.org/pdf/2306.02371.pdf) (CIKM 2023)

> studies have
found that the performance of dual-encoders are often limited due
to the neglecting of the interaction information between queries
and candidate passages.
...
recent state-of-the-art methods often introduce late-interaction during the model inference process. However,
such late-interaction based methods usually bring extensive computation and storage cost on large corpus. 
...
we Incorporate Implicit Interaction
into dual-encoders, and propose I
3
retriever. In particular, our implicit interaction paradigm leverages generated pseudo-queries to
simulate query-passage interaction, which jointly optimizes with
query and passage encoders in an end-to-end manner.

>  Unlike existing interaction schemes that requires
explicit query text as input, the implicit interaction is conducted
between a passage and the pseudo-query vectors generated from the
passage. Note that the generated pseudo-query vectors are implicit
(i.e., latent) without explicit textual interpretation. Such implicit
interaction paradigm is appealing, as 1) it is fully decoupled from
actual query, and thus allows high online efficiency with offline
caching of passage vectors, and 2) compared with using an off-theshelf generative model [[41]](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf) to explicitly generate textual pseudoquery, our pseudo-query is represented by latent vectors that are
jointly optimized with the dual-encoder backbone, which is more
expressive for the downstream retrieval task.

![I3 retriever](../../images/Screenshot 2023-11-02 at 16.17.53.png)
![I3 retriever](../../images/Screenshot 2023-11-02 at 16.27.27.png)


---
### [Multivariate Representation Learning for Information Retrieval](https://arxiv.org/pdf/2304.14522.pdf) (2023)

>  Instead of learning a vector for each query and
document, our framework learns a multivariate distribution and
uses negative multivariate KL divergence to compute the similarity
between distributions.
For simplicity and efficiency reasons, we
assume that the distributions are multivariate normals and then
train large language models to produce mean and variance vectors
for these distributions. 

> TBC

---
### [Scalable and Effective Generative Information Retrieval](https://arxiv.org/pdf/2311.09134.pdf) (2023)

> TBC

---
### [RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!](https://arxiv.org/pdf/2312.02724.pdf) (2023)

> TBC


---
### [Learning to Rank in Generative Retrieval](https://arxiv.org/pdf/2306.15222.pdf) (2023)

> LTRGR enables generative retrieval to learn to rank passages directly, optimizing
the autoregressive model toward the final passage ranking
target via a rank loss. This framework only requires an additional learning-to-rank training phase to enhance current
generative retrieval systems and does not add any burden
to the inference stage

> TBC


---
### [Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models](https://arxiv.org/pdf/2310.07712.pdf) (2023)

> TBC


---
### [LARGE LANGUAGE MODELS ARE EFFECTIVE TEXT RANKERS WITH PAIRWISE RANKING PROMPTING](https://arxiv.org/pdf/2306.17563.pdf) (2023)

> TBC


---
### [RankingGPT: Empowering Large Language Models in Text Ranking with Progressive Enhancement](https://arxiv.org/pdf/2311.16720.pdf) (2023)

> objective of LLMs, which typically centers
around next token prediction, and the objective
of evaluating query-document relevance. To address this gap and fully leverage LLM potential
in text ranking tasks, we propose a progressive
multi-stage training strategy. Firstly, we introduce a large-scale weakly supervised dataset of
relevance texts to enable the LLMs to acquire
the ability to predict relevant tokens without
altering their original training objective. Subsequently, we incorporate supervised training to
further enhance LLM ranking capability

> TBC


<!-- ---
### []() ()

> TBC-->


<!-- ---
### []() ()

> TBC-->


<!-- ---
### []() ()

> TBC-->

