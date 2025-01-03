---
layout: archive
title: "Retrieval Augmentation and In-Context Learning"
permalink: /readings/rag/
author_profile: false
sidebar: toc
redirect_from:
  - /readings/rag.html
---

---
### [Inference Scaling for Long-Context Retrieval Augmented Generation](https://arxiv.org/pdf/2410.04343) (2024-10-05)

*Google DeepMind*

> we investigate inference scaling
for retrieval augmented generation (RAG), exploring strategies beyond simply increasing the quantity of
knowledge. We focus on two inference scaling strategies: in-context learning and iterative prompting.

> increasing inference computation leads to nearly
linear gains in RAG performance when optimally allocated, a relationship we describe as the inference
scaling laws for RAG.

> Inspired by iterative
methods (Trivedi et al., 2023; Yoran et al., 2024), we develop iterative demonstration-based RAG
(IterDRAG). IterDRAG learns to decompose input queries into simpler sub-queries and answer them
using interleaved retrieval. By iteratively retrieving and generating upon sub-queries, LLMs construct
reasoning chains that bridge the compositionality gap for multi-hop queries.

> we measure computation by considering the total number of input tokens across all iterations,
referred to as the effective context length.

> We reverse the order of the
retrieved documents, placing higher-ranked documents closer to the query [(Liu et al., 2024b)](https://arxiv.org/pdf/2401.10225)


---
### [Memory-Augmented LLM Personalization with Short- and Long-Term Memory Coordination](https://arxiv.org/pdf/2309.11696.pdf) (2023)

> TBC

---
### [ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation](https://arxiv.org/pdf/2308.11131.pdf) (2023)

> TBC

---
### [RUEL: Retrieval-Augmented User Representation with Edge Browser Logs for Sequential Recommendation](https://paperswithcode.com/paper/ruel-retrieval-augmented-user-representation) (2023)

> we propose RUEL, a novel retrieval-based sequential recommender that can effectively incorporate external anonymous user behavior data.
...
We then design a contrastive learning framework with a momentum encoder and a memory bank to retrieve the most relevant and diverse browsing sequences from the full browsing log based on the semantic similarity between user representations. After retrieval, we apply an item-level attentive selector to filter out noisy items and generate refined sequence embeddings for the final predictor. 

> TBC

---
### [Atlas: Few-shot Learning with Retrieval Augmented Language Models](https://arxiv.org/pdf/2208.03299.pdf) (2022)

> Atlas retrieves relevant documents based on the current context by using a general-purpose dense retriever
using a dual-encoder architecture, based on the [Contriever (Izacard et al., 2022)](ltr.md#unsupervised-dense-information-retrieval-with-contrastive-learning-2022). The retrieved documents
are processed, along with the current context, by a sequence-to-sequence model using the Fusion-in-Decoder
architecture (Izacard & Grave, 2020) that generates the corresponding output. We study the impact of
different techniques to train Atlas on its few-shot performance on a range of downstream tasks, including
question answering and fact checking. We find that jointly pre-training the components is crucial for few-shot
performance.

> Retriever. Our retriever module is based on the [Contriever (Izacard et al., 2022)](ltr.md#unsupervised-dense-information-retrieval-with-contrastive-learning-2022), an information retrieval
technique based on continuous dense embeddings. The Contriever uses a dual-encoder architecture, where the
query and documents are embedded independently by a transformer encoder (Huang et al., 2013; Karpukhin
et al., 2020). Average pooling is applied over the outputs of the last layer to obtain one vector representation
per query or document. 

> Language model. For the language model, we rely on the T5 sequence-to-sequence architecture (Raffel
et al., 2019). We rely on the Fusion-in-Decoder modification of sequence-to-sequence models, and process
each document independently in the encoder.

> if the language model finds a document useful when generating the output, the retriever
objective should encourage the retriever to rank said document higher. This allows us to train models
using only query and output pairs from the task of interest, without relying on document annotations. 

> Attention Distillation (ADist). The first loss that we consider is based on the attention scores of the
language model, and is heavily inspired by Izacard & Grave (2021). The main idea is that the cross-attention
scores between the input documents and the output, can be used as a proxy of the importance of each input
document when generating the output.
...
average across heads, layers and tokens.
...
Here, this loss is only used to optimize the parameters of the retriever, and not the language model. When
using recent deep learning frameworks, this is achieved by applying a StopGradient operator on pattn.

> End-to-end training of Multi-Document Reader and Retriever (EMDR2). Next, we consider the
method introduced by [Sachan et al. (2021)](#end-to-end-training-of-multi-document-reader-and-retriever-for-open-domain-question-answering-2021), which is inspired by the expectation-maximization algorithm,
treating retrieved documents as latent variables.
...
Again, only the parameters of the retriever are updated by applying a StopGradient operator

> Perplexity Distillation (PDist). Third, ..., we want to train the retriever to predict how much each document would improve
the language model perplexity of the output, given the query.

> Leave-one-out Perplexity Distillation (LOOP). Finally, we propose an objective based on how much
worse the prediction of the language model gets, when removing one of the top-k retrieved documents.

> **Efficient retriever fine-tuning**: *Re-ranking.* A second strategy is to retrieve a larger number of documents L with the retriever, and to
re-embed and rerank these documents with the up-to-date retriever, and pass the resulting top-K to the
language model.

> *Query-side fine-tuning.* Finally, the last strategy is to decouple the encoding of the queries and documents.
In this case, we fix the parameters corresponding to the document encoder, and only train the parameters
corresponding to the query encoder. Thus, the embeddings of documents are fixed, and we do not need to
refresh the index, and thus there is no computational overhead. As we will see in practice, the impact of
fixing the documents encoder varies greatly for different tasks when a large training dataset is available. For
most of the few-shot settings that we consider, query-side finetuning does not have large performance impact,
and sometimes even slightly improves performance.

---
### [RAVEN: In-Context Learning with Retrieval Augmented Encoder-Decoder Language Models](https://arxiv.org/pdf/2308.07922.pdf) (2023)

> We
find that [ATLAS](#atlas-few-shot-learning-with-retrieval-augmented-language-models-2022) exhibits a certain in-context learning ability; however, due to a mismatch between
pretraining and testing and a limited context length—issues that are common to existing encoderdecoder LMs trained with masked language modeling—its few-shot performance is not stable and
providing more than, e.g., 8-shot, examples does not lead to further improvement.

> While there is growing interest in this area, most studies have focused on incontext learning with decoder-only LMs, e.g., GPT-3 (Brown et al., 2020). However, bidirectional
LMs like BERT (Devlin et al., 2019) and T5 (Raffel et al., 2020) have been shown to achieve
superior performance on various natural language understanding tasks, indicating that they may
offer unique advantages for in-context learning as well.
...
 For instance, Patel
et al. (2023) demonstrate that bidirectional models can outperform decoder-only LMs of a similar
scale regarding in-context learning; however, there is still a significant performance gap compared to
decoder-only models on a much larger scale.

> While there
has been some research on in-context learning with retrieval-augmented decoder-only LMs, which
can be straightforwardly implemented by concatenating retrieved passages with the query as the
input of the LM (Mallen et al., 2022; Shi et al., 2023; Khattab et al., 2022), in-context learning with
retrieval-augmented encoder-decoder LMs, such as ATLAS, remains unexplored

> TBC

---
### [RETRIEVAL MEETS LONG CONTEXT LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2310.03025.pdf) (2023)

> we find that LLM
with 4K context window using simple retrieval-augmentation at generation can
achieve comparable performance to finetuned LLM with 16K context window via
positional interpolation on long context tasks, while taking much less computation.
More importantly, we demonstrate that retrieval can significantly improve the
performance of LLMs regardless of their extended context window sizes

> TBC


---
### [In-Context Retrieval-Augmented Language Models](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/63c6c20dec4479564db21819_NEW_In_Context_Retrieval_Augmented_Language_Models.pdf) (2023)

> incontext RALM: leaving the LM architecture unchanged and prepending grounding documents
to the input. We show that in-context RALM
which uses off-the-shelf general purpose retrievers provides surprisingly large LM gains
across five diverse corpora.


---
### [MAKING RETRIEVAL-AUGMENTED LANGUAGE MODELS ROBUST TO IRRELEVANT CONTEXT](https://arxiv.org/pdf/2310.01558v1.pdf) (2023)

> TBC

---
### [UNDERSTANDING RETRIEVAL AUGMENTATION FOR LONG-FORM QUESTION ANSWERING](https://arxiv.org/pdf/2310.12150.pdf) (2023)

> TBC


---
### [RET-LLM: Towards a General Read-Write Memory for Large Language Models](https://arxiv.org/pdf/2305.14322.pdf) (2023)

> TBC


---
### [Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study](https://arxiv.org/pdf/2304.06762.pdf) (2023)

> TBC



---
### [RecMind: Large Language Model Powered Agent For Recommendation](https://arxiv.org/pdf/2308.14296.pdf) (2023)

> TBC


---
### [Generative Slate Recommendation with Reinforcement Learning](https://arxiv.org/pdf/2301.08632.pdf) (2023)

> TBC



---
### [LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking](https://arxiv.org/pdf/2311.02089.pdf) (2023)

> TBC


---
### [Integrating Summarization and Retrieval for Enhanced Personalization via Large Language Models](https://arxiv.org/pdf/2310.20081v1.pdf) (2023)

> TBC


---
### [End-to-End Training of Multi-Document Reader and Retriever for Open-Domain Question Answering](https://arxiv.org/pdf/2106.05346.pdf) (2021)

> TBC

---
### [Augmented Language Models: a Survey](https://arxiv.org/pdf/2302.07842.pdf) (2023)

> TBC

---
### [You can’t pick your neighbors, or can you? When and how to rely on retrieval in the kNN-LM](https://arxiv.org/pdf/2210.15859.pdf) (2022)

> we explore
the importance of lexical and semantic matching in the context of items retrieved by kNNLM. We find two trends: (1) the presence of
large overlapping n-grams between the datastore and evaluation set plays an important factor in strong performance, even when the datastore is derived from the training data; and (2)
the kNN-LM is most beneficial when retrieved
items have high semantic similarity with the
query.
![example](../../images/Screenshot 2024-01-17 at 19.12.49.png)
> TBC


---
### [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/pdf/2202.12837.pdf) (2022)

> TBC


---
### [Ground-Truth Labels Matter: A Deeper Look into Input-Label Demonstrations](https://arxiv.org/pdf/2205.12685.pdf) (2022)

> TBC


---
### [LARGER LANGUAGE MODELS DO IN-CONTEXT LEARNING DIFFERENTLY](https://arxiv.org/pdf/2303.03846.pdf) (2023)

> TBC


---
### [SAIL: Search-Augmented Instruction Learning](https://arxiv.org/pdf/2305.15225.pdf) (2023)

search engine --> duckduckgo

> TBC


---
### [RECITATION-AUGMENTED LANGUAGE MODELS](https://arxiv.org/pdf/2210.01296.pdf) (2023)

by Google research

> TBC


---
### [BlenderBot 3: a deployed conversational agent that continually learns to responsibly engage](https://arxiv.org/pdf/2208.03188.pdf) (2022)

by Meta AI -->  [Mojeek](https://www.mojeek.com/)

> TBC


---
### [Internet-Augmented Dialogue Generation](https://arxiv.org/pdf/2107.07566.pdf) (2021)

by Facebook AI research --> bing search api

> TBC


---
### [FRESHLLMS: REFRESHING LARGE LANGUAGE MODELS WITH SEARCH ENGINE AUGMENTATION](https://arxiv.org/pdf/2310.03214.pdf) (2023)

by Google and OpenAI

> TBC


---
### [Blinded by Generated Contexts: How Language Models Merge Generated and Retrieved Contexts for Open-Domain QA?](https://arxiv.org/pdf/2401.11911.pdf) (2023)

> we construct datasets with conflicting contexts, where each question is paired
with both generated and retrieved contexts, yet
only one of them contains the correct answer.
Our experiments reveal a significant bias in
LLMs (GPT-4/3.5 and Llama2) towards generated contexts, even when they provide incorrect information. We further identify two key
factors contributing to this bias: i) contexts generated by LLMs typically show greater similarity to the questions, increasing their likelihood
of selection; ii) the segmentation process used
in retrieved contexts disrupts their completeness, thereby hindering their full utilization in
LLMs.

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

<!-- ---
### []() ()

> TBC-->

<!-- ---
### []() ()

> TBC-->
