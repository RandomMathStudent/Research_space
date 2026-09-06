Decision: On the higher end of borderline accept - they found some insights, on a well chosen question, but communication and volume of output could have been better.



Measuring Faithfulness in Chain-of-Thought


Executive Summary
This work performs a comprehensive study on various factors affecting existing perturbation-type faithfulness test on Chain-of-Thought (CoT) - namely the mistake insertion test by Lanham. Faithfulness by the design of this test is defined as the change in prediction proceeding an erroneous CoT after a mistake is inserted into an intermediate reasoning step. I study GSM8K and evaluate the Gemma2 instruct-tuned series models of 2/9/27B sizes. Code is provided.

Faithfulness on task format
This finding shows that even within the same task, the model can have varying faithfulness levels when required to answer between MCQ or open-ended style. Figure 1 is read as gen: open-ended, f: front, b: back. Front refers to inserting the mistake at an earlier CoT step while back is at the later step (besides the last as it may directly change the answer). Figure 1 shows that MCQ format achieves much lower faithfulness even when the task is exactly the same. There exists a small difference in faithfulness between mistakes inserted at specific steps.

Figure 1. Faithfulness on open-ended vs MCQ format.

Faithfulness on samples requiring CoT
Figure 2 groups the faithfulness scores according to samples which achieves the same prediction between w and w/o CoT and different (diff) on MCQ-format. The finding shows that faithfulness is much higher when the model produces different predictions when generating CoT. 

Figure 2. Faithfulness on samples with similar/different predictions between w and w/o CoT on MCQ task

Self correction on CoT
I perform a check using GPT-4o on the possibility of the model self-correcting the mistake inserted to revert the CoT back to the original path. Correction rates are low (mostly < 10%), but however most represents cases of unfaithfulness, which should not be (false negatives). There is also a higher rate of correction detected in cases where the model gets a different answer with CoT.

Model
2B
9B
27B
Metric/Task-format
Gen
MCQ
Gen
MCQ
Gen
MCQ
Correction %
6
11
4
5
4
4
Unfaithful %
100
66
92
93
100
100
Diff prediction %
94
69
58
50
73
64

			

Neutral Faithfulness Test
Lower faithfulness scores on MCQ tasks may be related to the model opting for the same prediction as the closest approximation to the changed answer rather then not conditioning on the CoT. This test appends a neutral option: “None of the above” as an available choice. 

Figure 3. Faithfulness scores when a neutral option is given.

As shown in Figure 3, there is a marginal change in the faithfulness score despite given a plausible option here. 

Context-level analysis of CoT


Figure 4. Context-level attention scores of CoT


I performed a token-level analysis on the CoT to study the attention scores of the top 3 attention heads using logit attribution. Figure 4 (Gemma-2B) above shows the scores assigned to the three different input context: Few-shot (FS), input question (ques) and prior CoT. Both MCQ (M) and open-ended format shows that the model tends to attend a higher portion to the CoT context as compared to the rest, which may explain the higher levels of faithfulness attained when CoT is required. 







Introduction

There exist many forms of explanations available to interpret the inner workings of a Large Langauge Model (LLM). These can include circuit analysis such as the ones found in Wang to explain the behavior on Indirect Object Identification (IOI) tasks or using Sparse Autoencoder (SAE) to disentangle and sparsify highly active neurons as features to generate natural language descriptions. This work explores CoT as a form of natural language explanation and attempts to rigorously study popular faithfulness tests to ascertain the faithfulness of CoT to serve as explanations. The test in focus pertains to the insertion of a mistake in an intermediate CoT to see if the model changes its prediction. A change of prediction evaluates to the CoT being faithful. However, I show that there are multiple factors that can affect such an evaluation and this forms the main objective of this work.

Motivation

A recent study by Parcalabescu discusses the prevailing issues with existing CoT tests, mainly criticising that such tests mainly measures self-consistency: “model’s ability to produce a desirable output under the given perturbation” rather than actual faithfulness by Jacovi . nostalgebraist argued that an absence of prediction change does not necessarily constitute unfaithfulness, by introducing a possibility where the model is itself self-aware of the mistake introduced and performs some form of self-correction before responding with the same prediction. We perform a study on this and find that albeit such occurrences are rare, they do exist and account for a significant portion of unfaithfulness cases. Another argument against the unfaithfulness detected in such tests relates to instances where the model may not need the CoT to arrive at a decision, in which case the CoT merely acts as a transcript of the steps taken, thus making it less prone to changing its decision when the error is introduced.

Contributions
A study on how different answering format (MCQ or open-ended) can introduce variance into the faithfulness levels.
If a question is simple enough such that the same conclusion can be achieved without CoT, the model is less likely to alter its prediction given a mistake.
The self-correction possibility by nostalgebraist do happen but rarely, with higher occurrences seen in smaller models.
Introduce a new test adapted from mistakes by enabling the possibility of selecting a neutral option.
A mechanistic view on how CoT is dependent on different input context.





Experiments

Experiment Setting

I use 3 examples to act as few-shot to ensure a specific format of the CoT. This is to ensure that the mistakes introduced via GPT-4o affects reasoning steps that that are important towards the task. Since chat models are trained to response in a human-like manner, certain CoT sentences tend to be describing how the model intends to solve the problem without outlining any equations or solving for intermediate solutions. Inserting errors in such steps tends to not affect the prediction at all and lead to false negatives. Using few-shot also prevent excessively verbose CoT where a CoT might be truncated based on the pre-set generation budget. The mistake insertion and correctness check is also performed using designed 3-shot examples. Unless specified, the corruption is mainly introduced in the 1st step of the CoT, which we denote as “f” in Figure 1. 

1) Faithfulness on task format

While the work has shown that faithfulness levels are task-dependent. There hasn’t been an effort at studying the output format of the task at hand, since the tasks are all MCQ-based. Given that GSM8K is an open-ended task, i adapted the task to MCQ-style by appending three answer choices. The choices were augmented by applying a multiplier of 2,3 and the power of tens (ie if the answer is  within tens = add 10, hundreds = 100). 


Figure 1. Faithfulness on open-ended vs MCQ format.

Figure 1 shows that the model tends to achieve higher faithfulness levels on open-ended (gen). This may be less surprising since the metric function is more stringent on evaluating precise predictions as compared to picking out of a set of options. However, this introduces a limitation on the test itself: the evaluation varies according to how we evaluate the model rather than the actual faithfulness of the underlying CoT. The lower faithfulness in MCQ-style task may be due to the model reverting to the same prediction as the closest approximation which is difficult to see as compared to open-ended task.




2) Faithfulness on samples requiring CoT

A limitation with the mistake study by Lanham is that they did not account for the fact that certain samples may not require CoT at all to arrive at the decision. In other words, the answer predicted w and w/o CoT is the same. Though the paper did compare the accuracy  w and w/o CoT against faithfulness level, it is non-conclusive and furthermore the study should have been scrutinising the prediction change instead of accuracy scores. Thus, I group samples into same (when the prediction is same w and w/o CoT) and diff otherwise. 

Figure 2. Faithfulness on samples with similar/different predictions between w and w/o CoT on MCQ task

Figure 2 shows that same samples tend to achieve lower faithfulness scores as opposed to different. This supports the hypothesis by nostalgebraist, where the model may have come to a decision prior to the CoT and inserting a mistake has a lower tendency of altering the prediction. I observe lesser variance on generation task, mainly due to the high levels of faithfulness as compared to MCQ-style, albeit the findings are the same - different predictions have slightly higher levels of faithfulness. The experiments in section 1 and 2 show an interesting finding where 9B tends to achieve higher faithfulness when the mistake is inserted at the later step, inverse of the findings from Lanham.

3) Self-correction on CoT

While the mistake test perform checks at the answer level. It neglects the correctness of the subsequent CoT. nostalgebraist presented an interesting thought on the possibility of the model being “aware” of the mistake inserted and self-correcting it before arriving at the same prediction. However, under the mistake test, this would be marked as unfaithful, though it should not be. We perform a correctness check using GPT-4o with few-shot examples on the CoT continuation after the mistake insertion. The correctness in this case refers to correcting the CoT continuation towards the prediction rather than the correct answer, which is formatted in the few-shot examples.





Model
2B
9B
27B
Metric/Task-format
Gen
MCQ
Gen
MCQ
Gen
MCQ
Correction %
6
11
4
5
4
4
Unfaithful %
100
66
92
93
100
100
Diff prediction %
94
69
58
50
73
64


Though there are instances of correction, such occurrences are low especially on larger models. However, given that the majority of these cases represent unfaithful instances, this shows that scores achieved on the mistake test are possibly lower than what they represent. These correction instances also tend to happen on instances where the model arrives at a different prediction given the CoT. I perform a token-level analysis (explained in section 5) and find that on such instances, the model tends to attend more strongly to the question rather than prior CoT when predicting the corrected token (i manually annotated the small samples to find the exact corrected token). One hypothesis may be that on difficult samples, the model does not yet come to a conclusion while generating the CoT and is referencing the question while generating the subsequent tokens, thus there is a higher likelihood of correcting the erroneous CoT. 

4) Neutral Faithfulness Test

I introduced another faithfulness test, where instead of presenting a case of faithfulness as a change of prediction, the test instead considers a faithful event when the model chooses a neutral option when presented with an erroneous CoT. This is performed by adding a neutral option in the MCQ task: “None of the above”. In the former, when the model changes it’s decision, we are unclear if it does so because it is aware that the erroneous answer is closer to the changed option which it picks or if the behavior is broken due to the error introduced which lowers the probability of the original option. Moreover, just because the model changes it’s decision, it does not actually constitute faithfulness in a complete sense. Imagine a case where the neutral option is given, and the model picks a separate answer from the original option but not the neutral. In this case, it is extremely unlikely that the mistake leads to the counterfactual option. Though it does not prove that the original CoT is unfaithful, this test evaluates if the model is sufficiently self-aware that the reasoning steps it took did not arrive at any options. If the CoT truly leads to the original outcome and the model is aware of this (hence generating the CoT), then it should have picked the neutral option when it realises the CoT is corrupted. An inherent limitation is that this test may be hinged on the model’s ability to explore the neutral option. However, the goal of CoT as a form of explanation is to build trust in the user, and being sufficiently self-aware that it’s own explanation does not lead to a viable option is a reasonable expectation.

A concern of section 1 comparing the two task format is that the model trivally attains a lower level of faithfulness due to the picking the best possible option. However, as shown from Figure 3 in the summary, even when given a plausible option, it does not increase the faithfulness scores significantly (Figure 3 does not include samples for which the original prediction is the neutral option, thus picking the neutral option after the error should result in higher faithfulness). 

Under this test, the desired label is always the neutral option since even if the original prediction is neutral, the erroneous CoT is unlikely to result in any other option, thus the model would be faithful by sticking to the neutral label. Figure 5 compares the faithfulness as measured by picking any option so long as the prediction changes vs neutral after the mistake insertion.

Figure 5. Faithfulness on neutral vs normal MCQ task

It is observed that bigger model (9/27B) tend to select the neutral option more often than smaller model (2B), where the gap is bigger. 

5) Context-level analysis of CoT

In this experiment, I attempted to answer the question: which part of the input does the CoT rely on? In this case, the reliance is represented by observing the attention scores allocated to different parts of the context: few-shot examples (fs), question (ques) and prior CoT, thus i only measured the second CoT chain onwards. I study samples with 2-6 CoT which represents > 90% of the full experimental set. I select the top 3 attention heads via logit attribution on the CoT tokens. Since presumably not all tokens in the CoT are important, I filter down to tokens which represent digits given that the task involves primarily mathematics. Thus for each CoT token, the top three attention heads are selected based on the attribution score towards predicting the digit token and the attention scores are averaged across the 3 heads. The attention scores are then split into the respective context. 

Firstly as seen from Figure 4 in the summary section, the CoT’s are largely reliant on the prior CoT as compared to the question or few-shot, which may explain the high faithfulness scores achieved. The lower reliance on the question context may explain why the correction phenomenon occurs at a low rate. I observe similar proportions in the 9B model. I did not run this experiment on the 27B due to large memory requirements. 

Though the context proportions are not significantly different between task format. It differs when I look at the answer prediction. In Figure 6, when the model is evaluated under the MCQ format, the reliance on CoT is much lower as compared to the others. It is not definitive if this is one of the factors for the lower faithfulness attained in MCQ besides the aforementioned ones. I largely attribute this to the model attending to the answer options in the question to select the correct alphabet as compared to just producing a token not necessarily present in the question.


 Figure 6. Context-level attention scores of prediction

Next, I analyzed each CoT from the 2nd onwards on their attention score on the immediate previous CoT (different from the above where the prior CoT is the entire previous CoT). Seen in Figure 7, the scores decrease as the CoT advances towards the prediction. One hypothesis on samples with longer CoT, not all CoTs follow in a step-by-step manner, but rather certain CoT’s are listing down intermediate facts or products which are then aggregated in the final step to form the prediction.

Figure 7. Context-level attention scores of intermediate previous CoT on CoT generation
Key Takeaway

While CoT serves as an attractive form of interpretation, enabling the user to understand the inner workings of a model through natural language as compared to more complicated analysis such as circuits, measuring faithfulness of CoT is a difficult problem. One main source of difficulty arises from the lack of consensus on how faithfulness should be measured. Most existing works design specific tests to perturbate the CoT and evaluate if the affected output matches the designed outcome. As outlined in nostalgebraist, these test a CoT to be faithful based on the sufficiency of the CoT to act as the sole contributing factor towards the prediction and any modifications (truncate/mistake) to alter the prediction , following the specific conditional link: Input –> CoT –> answer. However as shown in section 2, the results from such test on samples which the model may not require the CoT may not be valid. Though Parcalabescu attempts to close the gap by instead ascertaining that a faithful CoT should have similar input attributions as the prediction, the limitation in their work is that the prediction is conditional on the CoT and comparisons made solely based on the input attributions violates the conditional dependency of the prediction on the CoT. 

This work has shown that even within a designed test, there can be multiple factors which introduces significant variance in the faithfulness measured. I proposed the neutral option test to serve as an improvement by checking for the correctness in the prediction change. The context-level analysis shows that CoT is largely dependent on the prior CoT, mimicking a step-by-step manner. Possible future directions could involve studying the alignment between context attribution against the natural language interpretation of the CoT itself, ie if the CoT involves describing a specific subject found in the question, then the input attribution towards that token should be large. Other directions could be to utilize SAE’s features on CoT to determine if the features correlate with the CoT, though it may be prone to the unfaithfulness of the SAE feature itself. 


Extras

The main rationale on using logit attribution with attention scores was that it was feasible as compared to other forms of analysis methods such as activation/path/attribution patching, knock-out/mean-ablation or even circuit analysis (though maybe overkill for token-level analysis). However the main issue that prevents the above mentioned techniques from being used was the applicability. Patching is primarily used for single-token analysis and for reasons mainly due to the strict requirements that come with it. The first requires the input space to be of similar length between the clean and corrupted sample. This is easy to do if one needs to only predict a single token. However when the target is a long sequence such as CoT, if one were to naively append the clean CoT token as prior context ( token 1 of clean CoT to predict token 2), this would essentially erase or dull the corruption effects introduced earlier, since later tokens are easier to predict given more clean tokens. Appending the corrupted CoT instead, would make it much harder to predict the clean tokens given the incoherence and increasing the total/indirect effects but for the wrong reasons. However if one does not require token-level analysis, and only attention/mlp layers on the last token, then the similar length requirement would not be an issue though the question of which token (clean/corrupt) to append would still be a problem. 

I am not sure if this is trivial but in the event where activation patching is performed only on the last token and across a large set of components (MLP, attn, residual of every layer), one could reduce the computation requirement by using KV caching on one sample and replicating the KV cache across a large batch. Thus you would only need to do 1 forward pass across the full context first for the KV cache up til the 2nd last token and then for the actual activation patching only do forward pass on the last token. ( this is a trade-off between memory and speed and i found that it significantly speeds up activation patching).
