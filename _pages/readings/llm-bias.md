---
layout: archive
title: "Bias in LLM Judgments"
permalink: /readings/llm-bias/
author_profile: false
sidebar: toc
redirect_from:
  - /readings/llm-bias.html
---

---
### [Judging the Judges: A Systematic Investigation of Position Bias in Pairwise Comparative Assessments by LLMs](https://arxiv.org/pdf/2406.07791) (31 Oct 2024)

> Our study introduces a systematic framework to examine position bias
in pairwise comparisons, focusing on repetition stability, position consistency,
and preference fairness.

> Our findings confirm that position bias in capable LLM judges is not
due to random chances, along with notable variations observed across judges and
tasks. Moreover, position bias is weakly influenced by the length of prompt components but significantly impacted by the quality gap between solutions. 

> Preference Fairness and Repetition Stability. Specifically, we move beyond
simply assessing Position Consistency by incorporating Preference Fairness, which provides deeper
insights into the specific answer directions where
models exhibit unfair preferences. Additionally,
the measurement of Repetition Stability ensures
that the observed position bias in the given model
and tasks is not due to random variations, thus
strengthening the reliability of the findings.

> Repetition Stability (RS) evaluates the reliability of LLM judges when presented with identical
queries multiple times. 

> Position Consistency (PC) quantifies how frequently a judge model prefers the same solution
after the order of solutions is permuted.

> Preference Fairness (PF) measures the extent
to which judge models favor certain solution positions.

> Instances where numerous LLMs
agree are generally easier to judge, whereas instances with disagreements are more challenging to evaluate and more prone to position bias.

> Future work could explore
how to measure the likelihood of position bias arise
from the datasets by identifying and quantifying
such hard-to-judge instances before implementing
LLM judges.

> we measure the answer quality gap by the win rates of candidates over
an expected baseline on a set of tasks and questions.

> The judges achieving close-to 0 PF in general, such as GPT-4 and Claude-3.5-Sonnet, exhibit varied preference directions across
tasks, preferring primacy on some tasks while recency on others. Particularly, o1-mini, while being
primacy-preferred on coding, extraction, and math,
exhibits almost fair preferences on reasoning, role
play, and writing tasks. Even for judges that are
recency-preferred across all tasks (e.g., Claude-3’s
and Gemini-pro’s), the extent of biased preference,
as reflected by P F values, varies by task.

> Moreover, a high position consistency does not
guarantee fairness. For example, on coding task
evaluations, GPT-4 and GPT-4o achieve the top
consistency but are significantly recency-preferred
and primacy-preferred, respectively. In comparison, GPT-3.5-Turbo is highly preference fair while
having comparable consistency.

> more capable models, such as GPT-4o
and Claude-3.5-Sonnet, maintain high consistency
when transitioning from pairwise to list-wise evaluations, while less capable models, such as GPT-3.5-
Turbo, exhibit greater sensitivity to the increased
number of candidates in list-wise tasks.

> the
most challenging instances to evaluate are characterized by: (1) frequent disagreements among LLM
judges, (2) closely matched win rates and minimal
quality gaps among candidate models, and (3) significant position bias exhibited by the majority of
judges. 


---
### [Split and Merge: Aligning Position Biases in LLM-based Evaluators](https://aclanthology.org/2024.emnlp-main.621.pdf) (2024.emnlp-main)

> we propose PORTIA, an alignmentbased system designed to mimic human comparison strategies to calibrate position bias in a
lightweight yet effective manner. Specifically,
PORTIA splits the answers into multiple segments, taking into account both length and semantics, and merges them back into a single
prompt for evaluation by LLMs.

![split and merge](../../images/Screenshot 2024-12-24 at 11.54.53.png)


---
### [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting](https://arxiv.org/pdf/2306.17563) (NAACL 2024)

by Google

> We analyze pointwise and listwise ranking prompts used by existing methods and argue that off-the-shelf LLMs do not
fully understand these challenging ranking formulations. In this paper, we propose to significantly reduce the burden on LLMs by using a new technique called Pairwise Ranking
Prompting (PRP).

> Since it is known that LLMs can be sensitive
to text orders in the prompt (Lu et al., 2022; Liu
et al., 2023a), for each pair of documents, we
will inquire the LLM twice by swapping their order: u(q, d1, d2) and u(q, d2, d1). Such simple debiasing method is difficult for listwise methods due
to their combinatorial nature.

> We introduce a sliding window approach that is
able to further bring down the computation complexity. One sliding window pass is similar to one
pass in the Bubble Sort algorithm: Given an initial
ranking, we start from the bottom of the list, compare and swap document pairs with a stride of 1
on-the-fly based on LLM outputs. One pass only
requires O(N) time complexity. See Figure 3 for
an illustration.
By noticing that ranking usually only cares about
Top-K ranking metrics, we can perform K passes,
where K is small

---
### [Self-Preference Bias in LLM-as-a-Judge](https://arxiv.org/pdf/2410.21819) (29 Oct 2024)

> Our findings reveal that LLMs assign significantly higher evaluations to outputs with lower perplexity than human evaluators, regardless of whether the outputs were self-generated. This suggests that the essence of the bias lies in perplexity and that the self-preference bias exists because LLMs prefer texts more familiar to them.

---
### [A Survey on LLM-as-a-Judge](https://arxiv.org/pdf/2411.15594) (23 Nov 2024)

> TBC


---
### [Can We Instruct LLMs to Compensate for Position Bias?](https://aclanthology.org/2024.findings-emnlp.732.pdf) (2024.findings-emnlp)

> In this work, we examine how to direct LLMs to allocate more attention towards a
selected segment of the context through prompting, aiming to compensate for the shortage of
attention. We find that language models do not
have relative position awareness of the context
but can be directed by promoting instruction
with an exact document index. 

---
### [From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge](https://arxiv.org/pdf/2411.16594) (25 Nov 2024)

> TBC


---
### [Large Language Models Are Not Robust Multiple Choice Selectors](https://arxiv.org/pdf/2309.03882) (7 Sep 2023 - ICLR 2024 Spotlight)

> Through extensive empirical analyses with 20 LLMs on three benchmarks, we pinpoint that this behavioral bias primarily stems from LLMs’ token
bias, where the model a priori assigns more probabilistic mass to specific option
ID tokens (e.g., A/B/C/D) when predicting answers from the option IDs. To mitigate selection bias, we propose a label-free, inference-time debiasing method,
called PriDe, which separates the model’s prior bias for option IDs from the overall prediction distribution. PriDe first estimates the prior by permutating option
contents on a small number of test samples, and then applies the estimated prior
to debias the remaining samples. We demonstrate that it achieves interpretable
and transferable debiasing with high computational efficiency. 

> we find that, contrary to the common view in previous work (Wang et al., 2023a; Pezeshkpour & Hruschka, 2023),
selection bias arises less from LLMs’ position bias, where they are deemed to favor options presented at specific ordering positions (like first or last). In contrast, we pinpoint one more salient
intrinsic cause of selection bias as the model’s token bias when predicting answers from the option
IDs given the standard MCQ prompt, where the model a priori assigns more probabilistic mass to
specific ID tokens (e.g., A/B/C/D).

> Despite the notably reduced selection bias, we find that removing option IDs usually degrades model
performance (except in a few cases under the 5-shot setting), see Table 3 and 4 in Appendix C. This
performance degradation results from the way we leverage LLMs to answer MCQs without option
IDs, i.e., calculating and comparing the likelihoods of options, which is referred to as the “cloze
prompt” format in Robinson & Wingate (2022). Their study demonstrates that asking LLMs to
predict option IDs forms a better MCQ prompt than the “cloze prompt”, which is consistent with
our observation

> selection bias is an inherent behavioral
bias of LLMs that cannot be addressed by simple prompt engineering.

> The core idea of our method PriDe is to obtain a debiased prediction distribution by separating the
model’s prior bias for option IDs from the overall prediction distribution.

> Since gpt-3.5-turbo does not return the output probability, we sample 100
generated answers as an approximation to Pobserved



---
### [Bias in Large Language Models: Origin, Evaluation, and Mitigation](https://arxiv.org/pdf/2411.10915) (16 Nov 2024)

> Prompting methods gain popularity since the general public has no access to the model’s internal structure due to business
interest. Specifically, [Li et al. (2024a)](#steering-llms-towards-unbiased-responses-a-causality-guided-debiasing-framework-13-mar-2024) proposes causal prompting based on front-door adjustment
(Pearl et al., 2016). The proposed method modifies prompts without access to the parameters and
logits of LLMs. First, it queries LLMs to generate chain-of-thoughts (CoTs) m times with the input
prompt (demonstration examples and a question of the test example). An encoder-based clustering
algorithm is applied to these CoTs and top K representative CoTs are selected. Next, it retrieves
the optimal demonstration examples for each representative. Finally, LLMs are queried T times to
obtain T answers for each representative CoT, and the final answer is obtained by a weighted voting.


---
### [Steering LLMs Towards Unbiased Responses: A Causality-Guided Debiasing Framework](https://arxiv.org/pdf/2403.08743) (13 Mar 2024)

> TBC

---
### [Debating with More Persuasive LLMs Leads to More Truthful Answers](https://arxiv.org/pdf/2402.06782) (9 Feb 2024)

> we ask: can weaker models assess the correctness of stronger models? We
investigate this question in an analogous setting,
where stronger models (experts) possess the necessary information to answer questions and weaker
models (non-experts) lack this information but are
otherwise as capable. The method we evaluate is
debate, where two LLM experts each argue for a
different answer, and a non-expert selects the answer. On the QuALITY comprehension task, we
find that debate consistently helps both non-expert
models and humans answer questions, achieving
76% and 88% accuracy respectively (naive baselines obtain 48% and 60%). Furthermore, optimising expert debaters for persuasiveness in an
unsupervised manner improves non-expert ability to identify the truth in debates.
![prompt](../../images/Screenshot 2024-12-27 at 14.06.49.png)

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
