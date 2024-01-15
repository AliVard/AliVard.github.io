---
layout: archive
title: "LLM for Recommendation"
permalink: /readings/rec/
author_profile: false
sidebar: toc
redirect_from:
  - /readings/rec.html
---


---
### [PALR: Personalization Aware LLMs for Recommendation](https://arxiv.org/pdf/2305.07622.pdf) (2023 - short)

> we first use user/item
interactions as guidance for candidate retrieval, and then adopt an
LLM-based ranking model to generate recommended items.
... we fine-tune an LLM of 7 billion parameters
for the ranking purpose. This model takes retrieval candidates in
natural language format as input, with instructions explicitly asking
to select results from input candidates during inference. 

> Initially, we use an LLM and user
behavior as input to generate user profile keywords. Then, we
employ a retrieval module to pre-filter some candidates from the
items pool based on the user profile.
... Finally, we use LLM
to provide recommendations from those candidates based on user
history behaviors. To better adapt these general-purpose LLMs to
fit the recommendation scenarios, we convert user behavior data
into natural language prompts and fine-tune a LLaMa[22] 7B model.
Our goal is to teach the model to learn the co-occurrence of user
engaged item patterns. This approach enables us to incorporate user
behavior data into the LLM’ reasoning process and better generalize
to new users and unseen items. 

> An LLM can be
leveraged to generate a summary of a user’s preferences.
The "Natural Language Prompt" for the "LLM for
recommendation" comprises three components: the "Interaction History Sequence," the "Natural Language User Profile,"
and the "Candidates for Recommendation". The "Interaction History Sequence" is created by simply concatenating
the items that the user has interacted with. The "Natural
Language User Profile" is a high-level summarization of the
user’s preferences, generated using an LLM based on useritem interactions, item attributes, or even user information
if possible. The "Candidates for Recommendation" are the
output of a retrieval model, and in our design, we have the
flexibility to use any retrieval model for this purpose.

Finetune:
> The "Recommend" task involves a list of items that the user has interacted with in the past
(with a maximum limit of 20 items), and the objective of the model
is to generate a list of "future" items that the user may interact with.
We refer to a model fine-tuned by this instruction as 𝑃𝐴𝐿𝑅𝑣1.

> The "Recommend_Retrieval" task asks the model to retrieve the
target "future" items from a list of candidate items. The candidate
list contains all target items, plus a few negative items similar to the
target items (e.g. movies with the same genres, co-watched by many
users). We refer to a model fine-tuned with both "Recommend"
and "Recommend_Retrieval" instruction as 𝑃𝐴𝐿𝑅𝑣2.

> In this paper, we utilize
SASRec as our retrieval layer and consider its top 50 recommendations. By comparing 𝑃𝐴𝐿𝑅𝑣2 and SASRec, it’s obvious that the
top10 recommendations re-ranked by our PALR are superior to
the original recommendations provided by SASRec.

> We could observe
𝑃𝐴𝐿𝑅𝑣1 has shown some ability to connect historical interacted
items with possible future interacted items. Prior to fine-tuning, the
model tends to only recommend popular movies in movie recommendation tasks. However, 𝑃𝐴𝐿𝑅𝑣1 isn’t able to retrieve the target
item from a list of candidate items. We have tried to use 𝑃𝐴𝐿𝑅𝑣1
for retrieval and observe that it could only randomly select from
the candidates. The performance from 𝑃𝐴𝐿𝑅𝑣2 has demonstrated
the effectiveness of incorporating an additional instruction during
the fine-tuning stage.


---
### [Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction](https://arxiv.org/pdf/2305.06474.pdf) (2023)

> find that zero-shot LLMs lag behind traditional recommender models that have the access to user
interaction data, indicating the importance of user interaction data. However, through fine-tuning, LLMs achieve comparable or even
better performance with only a small fraction of the training data, demonstrating their potential through data efficiency.

> The rating prediction task could be formulated into one of two tasks: (1)
multi-class classification [(5 classes)]; or (2) regression [(single output)].

> For fine-tuning methods, we use Flan-T5-Base (250M) and Flan-T5-XXL (11B) models in
the experiments. We set the learning rate to 5e-5, batch size to 64, drop out rate to 0.1 and train 50k steps on all datasets.

---
### [Large Language Model Augmented Narrative Driven Recommendations](https://arxiv.org/pdf/2306.02250.pdf) (2023)

> We use LLMs for authoring synthetic
narrative queries from user-item interactions with few-shot prompting and train retrieval models for NDR on synthetic queries and
user-item interaction data. 

> given a user’s interactions, 𝐷𝑢, with items and their accompanying text documents (e.g., reviews,
descriptions) 𝐷𝑢, selected from a user-item interaction dataset I, we prompt InstructGPT, a 175B parameter
LLM, to author a synthetic narrative query 𝑞𝑢 based on 𝐷u.

> we only retain some of the items present in {𝑑𝑖 }
before
using it for training retrieval models. For this, we use a pre-trained language model to compute the likelihood of the
query given each user item, 𝑃𝐿𝑀 (𝑞𝑢 |𝑑𝑖), and only retain the top 𝑀 highly scoring item for 𝑞𝑢. In our experiments, we use FlanT5 with 3B parameters for computing and follow Sachan et al. [40] for computing 𝑃𝐿𝑀 (𝑞𝑢 |𝑑𝑖).



---
### [Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models](https://arxiv.org/pdf/2306.10933.pdf) (2023)

> we propose
an Open-World Knowledge Augmented Recommendation Framework with Large Language Models, dubbed KAR, to acquire two
types of external knowledge from LLMs — the reasoning knowledge on user preferences and the factual knowledge on items.
We
introduce factorization prompting to elicit accurate reasoning on
user preferences. The generated reasoning and factual knowledge
are effectively transformed and condensed into augmented vectors by a hybrid-expert adaptor in order to be compatible with the
recommendation task. The obtained vectors can then be directly
used to enhance the performance of any recommendation model.
We also ensure efficient inference by preprocessing and prestoring
the knowledge from the LLM.

>  the results of using LLMs as recommenders are far
from optimal for real-world recommendation scenarios

> Compositional gap. LLMs often
suffer from the issue of compositional gap, where LLMs have difficulty in generating correct answers to the compositional problem
like recommending items to users, whereas they can correctly answer all its sub-problems [52](glm.md#measuring-and-narrowing-the-compositionality-gap-in-language-models-2023). 

> KAR consists of three stages: (1) knowledge reasoning and generation, (2) knowledge adaptation, and (3) knowledge
utilization. For knowledge reasoning and generation, to avoid the
compositional gap, we propose factorization prompting to break
down the complex preference reasoning problem into several key
factors to generate the reasoning knowledge on users and the factual knowledge on items. Then, the knowledge adaptation stage
transforms the generated knowledge to augmented vectors in recommendation space. In this stage, we propose hybrid-expert adaptor
module to reduce dimensionality and ensemble multiple experts for
robust knowledge learning, thus increasing the reliability and availability of the generated knowledge. Finally, in the knowledge utilization stage, the recommendation model incorporates the augmented
vectors with original domain features for prediction, ...

> provide extra knowledge for classical recommendations for better user or item representations.
ZEREC [9 (2021)](https://arxiv.org/pdf/2105.08318.pdf) incorporates traditional recommender systems with
PLMs to generalize from a single training dataset to other unseen
testing domains. UniSRec [22 (2022)](https://arxiv.org/pdf/2206.05941.pdf) utilizes BERT to encode user
behaviors, and therefore learns universal sequence representations
for downstream recommendation. Built upon UniSRec, VQ-Rec [21 (2023)](https://arxiv.org/pdf/2210.12316.pdf)
further adopts vector quantization techniques to map language embeddings into discrete codes, balancing the semantic knowledge
and domain features.
LLM-Rec [44 (2023)](#llm-rec-personalized-recommendation-via-prompting-large-language-models-2023) investigates various prompting strategies to generate
augmented input text from GPT-3 (text-davinci-003), which improves the recommendation capabilities. Another work [45 (2023)](#large-language-model-augmented-narrative-driven-recommendations-2023) utilizes
InstructGPT (175B) [48 (2022)](https://arxiv.org/pdf/2203.02155.pdf) for authoring synthetic narrative queries
from user-item interactions and train retrieval models for narrativedriven recommendation on synthetic data. TagGPT [33 (2023)](https://arxiv.org/pdf/2304.03022.pdf) provides a
system-level solution of tag extraction and multi-modal tagging in a
zero-shot fashion equipped with GPT-3.5 (gpt-3.5-turbo).

> **Knowledge Reasoning and Generation.**
When the request to an LLM is too general, the generated factual knowledge may be correct but useless, as it may not align with the inferred user
preferences.
With the factors incorporated into preference reasoning prompt, the complex
preference reasoning problem can be broken down into simpler
subproblems for each factor, thus alleviating the compositional gap
of LLMs.
*(1) Scenario-specific Factors.* interactive collaboration with LLMs and expert opinions.
*(2) LLM as Preference Reasoner & Knowledge Provider.* Preference reasoning prompt is constructed with the user’s
profile description, behavior history, and scenario-specific factors.
Item factual prompt is designed to fill the knowledge gap
between the candidate items and the generated reasoning knowledge.

> **Knowledge Adaptation.**
two modules: The knowledge
encoder module encodes the generated textual knowledge into
dense vectors and aggregates them effectively (average pooling). The hybrid-expert
adaptor converts dense vectors from the semantic space to the
recommendation space. It tackles dimensionality mismatching and
allows for noise mitigation.

> **Knowledge Utilization.**
these augmented vectors are directly treated as
additional input features. Specifically, we use them as additional
feature fields in recommendation models, allowing them to explicitly interact with other features. During training, the hybrid-expert
adaptor module is jointly optimized with the backbone model to
ensure that the transformation process adapts to the current data
distribution. 

> *Implementation Details.* We utilize API of a widely-used
LLM for generating reasoning and factual knowledge. Then, ChatGLM [10] is employed to encode the knowledge, followed by average pooling as the aggregation function in Eq. (1). Each expert in
the hybrid-expert adaptor is implemented as an MLP with a hidden
layer size of [128, 32]. The number of experts varies slightly across
different backbone models, typically with 2-5 shared experts and
2-6 dedicated experts.

> For
example, when using FiBiNet as the backbone model on MovieLens1M, KAR achieves a 1.49% increase in AUC and a 2.27% decrease
in LogLoss.

> KAR shows
more remarkable improvement in feature interaction models compared to user behavior models. 

> For example, when PRM is employed as the backbone, KAR achieves
a remarkable increase of 5.71% and 4.71% in MAP@7 and NDCG@7.

> we observe that both reasoning knowledge and
factual knowledge can improve the performance of backbone models, with reasoning knowledge exhibiting a larger improvement.
This could be attributed to the fact that reasoning knowledge inferred by the LLMs captures in-depth user preferences, thus compensating for the backbone model’s limitations in reasoning underlying motives and intentions. Additionally, the joint use of both
reasoning and factual enhancements outperforms using either one
alone, even achieving a synergistic effect where 1 + 1 > 2. One
possible explanation is that reasoning knowledge contains external
information that is not explicitly present in the raw data. When
used independently, this external knowledge could not be matched
with candidate items. However, combining the externally generated
factual knowledge on items from the LLMs aligned with reasoning knowledge allows RSs to gain a better understanding of items
according to user preferences.

---
### [LLM-Rec: Personalized Recommendation via Prompting Large Language Models](https://arxiv.org/pdf/2307.15780.pdf) (2023)

- It seems that this paper only used *content description* and not collaborating signals.
- Got rejected from ICLR 2024.

> Our empirical experiments show that incorporating the augmented input text generated by LLM leads to improved recommendation performance. Recommendation-driven and engagement-guided prompting strategies
are found to elicit LLM’s understanding of global and local item characteristics.

> Rather than using LLMs as
recommender models, this study delves into the exploration of prompting strategies to augment input
text with LLMs for personalized content recommendation. 

> To create the engagement-guided prompt, we combine the content description of the target item,
denoted as dtarget, with the content descriptions of T important neighbor items, represented as
d1, d2, · · · , dT . The importance is measured based on user engagement. We will discuss more
details in the Experiment section. This fusion of information forms the basis of the prompt, which
is designed to leverage user engagement and preferences in generating more contextually relevant
content descriptions: “Summarize the commonalities among the following descriptions: ‘dtarget’; ‘d1; d2; ... dT ’.”

> Item module. Text encoder: We use Sentence-BERT [14] to derive the textual embeddings from the original content description and augmented text. The embedding model is all-MiniLM-L6-v2.

> Importance measurement for engagement-guided prompting: In our study, we show
an example of use Personalized PageRank (PPR) score as the importance measurement.
... The rationale
behind this approach lies in the observation that when users frequently engage with two
items, there tends to be a greater similarity in terms of user preferences. 

> LLM-Rec
boosts simple MLP models to achieve superior recommendation performance, surpassing other more
complex feature-based recommendation methods.

---
### [Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065.pdf) (2023)


> we
create semantically meaningful tuple of codewords to serve as a Semantic ID for
each item. 

> To do this, instead of assigning randomly generated atomic IDs to each item, we generate Semantic IDs: a semantically meaningful tuple of codewords for each item that serves as its unique identifier. We use a hierarchical method called RQ-VAE to generate these codewords. Once we have the Semantic IDs for all the items, a Transformer based sequence-to-sequence model is trained to predict the Semantic ID of the next item. Since this model predicts the tuple of codewords identifying the next item directly in an autoregressive manner, it can be considered a generative retrieval model. 

> TIGER is characterized by a new approach to represent each item by a novel "Semantic ID": a sequence of tokens based on the content information about the item (such as its text description). Concretely, given an item’s text description, we can use pre-trained text encoders (e.g., SentenceT5 [25]) to generate dense content embeddings. A quantization scheme can then be applied over the embeddings to form a small set of tokens/codewords (integers). We refer to this ordered tuple of codewords as the Semantic ID of the item. 

> we are the first to use generative Semantic IDs created using an auto-encoder (RQ-VAE [20, 45]) for retrieval models.

---
### [Augmenting the User-Item Graph with Textual Similarity Models](https://arxiv.org/pdf/2109.09358.pdf) (2021)

> A paraphrase similarity model
is applied to widely available textual data – such as reviews and
product descriptions – yielding new semantic relations that are
added to the user-item graph. This increases the density of the
graph without needing further labeled data.

> we complement the implicit item similarity
learnt from interactions by introducing explicit semantic relations
based on textual attributes.


---
### [Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)](https://arxiv.org/pdf/2203.13366.pdf) (2023)

> In P5, all data such as user-item interactions, user descriptions, item metadata, and user reviews are converted to a
common format — natural language sequences. The rich information from natural language assists P5 to capture deeper semantics
for personalization and recommendation. Specifically, P5 learns
different tasks with the same language modeling objective during
pretraining. 

> With adaptive personalized prompt for different
users, P5 is able to make predictions in a zero-shot or few-shot
manner and largely reduces the necessity for extensive fine-tuning.

> The
collection covers five different task families – rating, sequential
recommendation, explanation, review, and direct recommendation. 

> to make P5 aware of the personalized information
contained in the input sequence, we also apply whole-word embeddings W to indicate whether consecutive sub-word tokens are
from the same original word.
---
### [Large Language Models are Competitive Near Cold-start Recommenders for Language- and Item-based Preferences](https://arxiv.org/pdf/2307.14225.pdf) (2023)

> TBC

---
### [User Embedding Model for Personalized Language Prompting](https://arxiv.org/pdf/2401.04858.pdf) (2024)

> we introduce a new User Embedding Module (UEM) that efficiently processes
user history in free-form text by compressing
and representing them as embeddings, to use
them as soft prompts to a LM.

> Recent research has predominantly concentrated
on examining smaller segments of user history by
selecting representative samples from a users’ history (Salemi et al., 2023). 

> we employ an embedding-based technique to compress
the user’s entire history, creating a sequence of
representative user embedding tokens.
... Further, since the User Embedding Module (UEM) module is co-trained with the LM, the
representations are learned in-context for the specific tasks. ... 
Compared to the naive approach of concatenating user history and incurring O(n^2) compute
cost for self-attention, our approach demonstrates
a cheap way to incorporate history metadata as an
embedding thus dramatically reducing the required
compute.

<!-- ---
### []() ()

> TBC -->


<!-- ---
### []() ()

> TBC -->


<!-- ---
### []() ()

> TBC -->


<!-- ---
### []() ()

> TBC -->

