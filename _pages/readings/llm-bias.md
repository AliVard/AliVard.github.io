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
