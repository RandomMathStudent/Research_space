    # Reducing Prompt-Injection Following with a Tool-Token Role Vector

## Executive summary

I tested whether a residual-stream intervention can make an open-weight model less likely to treat instructions embedded in tool output as instructions. The intervention subtracts a direction that distinguishes user-role from tool-role activations, and applies it only to tokens belonging to the tool result.

The strongest experiment used 633 injected prompts per condition. Under a response-grounded behavioral classifier, clear credential-reconnaissance or exfiltration attempts fell from 57/633 (9.00%) with no intervention to 25/633 (3.95%) when the role vector was applied at layer 12, and 23/633 (3.63%) when it was applied at a layer selected to match the prompt. This is a reduction of 5.06 and 5.37 percentage points respectively.

These results are consistent with the role-confusion hypothesis, but they do not establish it. The confirmatory run contains only three clean controls per condition and does not save a benign-tool-use label, so it cannot by itself show that the reduction in injection following preserves ordinary tool use. An earlier 6,600-trial single-layer sweep did record benign-tool-use success, but did not show a corresponding behavioral injection reduction under the later response-grounded classifier. The mechanistic and behavioral findings should therefore be reported separately and cautiously.

## Question and hypothesis

Tool output should be interpreted as untrusted data. Prompt injection occurs when text in that output nevertheless changes the model's behavior as though it were an authorized instruction. I investigate whether this failure partly reflects an internal confusion between the user and tool roles.

For controlled text placed in user and tool roles, define a role direction at layer $\ell$ as

$$
v_{\mathrm{user-tool}} = \mu_{\mathrm{user}} - \mu_{\mathrm{tool}}.
$$

For every token position $i$ in the tool result, the intervention is

$$
h'_{\ell,i} = h_{\ell,i} - \alpha \frac{v_{\mathrm{user-tool}}}{\lVert v_{\mathrm{user-tool}} \rVert}.
$$

The behavioral hypothesis is that this shift reduces the probability that the model follows an injected instruction in tool output. The original preregistered-style success criterion was stronger: the role vector should improve over both no intervention and an equal-norm random-vector control while keeping benign tool use near baseline.

## Experimental setup

The experiments use an open-weight `gpt-oss` model in a simulated tool-use setting. Each prompt contains a legitimate user task and a tool result. Injected trials place a harmless proxy for malicious instruction following inside that result; the audit classifier additionally treats observable credential-search and exfiltration-style tool calls as clear behavioral failures. No real credentials, shell execution, or external upload are used.

Before generation, the complete chat prompt is rendered and the tool-result token positions are identified. A forward hook modifies only those residual-stream positions. The principal controls are a no-intervention baseline and, in the exploratory sweeps, an equal-norm random vector. Each raw record saves the rendered prompt, tool content, response, intervention condition, and token-span metadata for later review.

### Behavioral labels

The first analyses used a marker-based label: any mention of an injected marker counted as attack following. This was too broad, because a model can quote or discuss the injected text without acting on it. I therefore reclassified saved responses using a transparent response-level scheme:

| Category | Clear behavioral failure? |
| --- | --- |
| Credential reconnaissance attempt | Yes |
| Exfiltration attempt | Yes |
| Generic tool call | No; retained for review |
| Refusal or safe discussion | No |
| No observable injection action | No |

This classifier is intentionally conservative. It is useful for distinguishing observable action from mention of an attack string, but it remains a heuristic and requires manual transcript review before stronger claims.

## Experiments and results

### Exploratory fixed-layer sweep

The earlier single-layer experiment applied the intervention at layer 12 to 6,600 injected trials: 600 prompts in each of baseline, five random-vector magnitudes, and five role-vector magnitudes. It also included 33 clean controls.

The intervention produced the expected mechanistic movement. At $\alpha=5$, mean injection-token Userness shifted by -0.1539 for the role vector, compared with +0.0103 for the random-vector control. The same intervention shifted benign tool-result Userness by -0.1458, compared with +0.0125 for the random control. Thus the vector strongly changed the role-probe measurement in the intended direction.

This mechanistic effect did not yield a clear behavioral advantage in the reclassification. Clear behavioral-failure rates were 9.83% at baseline, 8.50% for the random vector at $\alpha=5$, and 7.50% for the role vector at $\alpha=5$ (600 responses in each condition). The benign-tool-use rates saved during this sweep were 81.33%, 81.50%, and 81.00% respectively. The observed behavioral differences are modest and this sweep should be treated as exploratory.

### Layer-matched exploratory comparison

A later 750-response comparison evaluated baseline plus layer-matched role-vector and random-vector conditions. Its response-grounded failure rates were 6.00% for baseline, 4.00% and 2.67% for layer-matched role-vector magnitudes $\alpha=4$ and $\alpha=5$, and 5.33% for both random-vector magnitudes. The small sample and multiple conditions make this supporting evidence rather than a final estimate.

### Clean-full comparison

The clean-full follow-up fixed the magnitude at $\alpha=5$ and ran 633 injected prompts in each of three conditions:

| Condition | Clear behavioral failures | Rate | Change from baseline |
| --- | ---: | ---: | ---: |
| Baseline | 57 / 633 | 9.00% | 0.00 pp |
| Fixed layer-12 role vector | 25 / 633 | 3.95% | -5.06 pp |
| Layer-matched role vector | 23 / 633 | 3.63% | -5.37 pp |

Generic tool-call rates were low and similar across these conditions: 2.05%, 1.74%, and 1.90%. Mean generated-response lengths were 466, 412, and 506 tokens, while maximum-token termination rates were 15.0%, 11.8%, and 15.6%. These diagnostics do not point to a uniform collapse in generation, although they are not a substitute for a proper utility evaluation.

## Interpretation

Taken together, the experiments give preliminary evidence that a role-derived intervention on tool tokens can reduce observable prompt-injection-following behavior. The result is more compelling in the clean-full comparison than in the initial fixed-layer sweep. One possible explanation is that matching the intervention layer to the prompt makes the role signal more effective, but the fixed layer-12 condition also improves substantially in the clean-full run, so this experiment does not isolate that explanation.

The single-layer sweep also establishes a useful dissociation: the role vector reliably reduces a Userness probe on both injected and benign tool tokens, but probe movement alone does not guarantee a comparably large behavioral result. A role probe is therefore a diagnostic measurement, not a sufficient mechanism-level explanation.

## Limitations

- The behavioral classifier is pattern-based and may miss or misclassify actions; full transcripts and ambiguous cases need human review.
- The clean-full comparison has no equal-norm random-vector condition. It cannot rule out that a comparably sized non-role perturbation would achieve some of the observed reduction.
- The clean-full run has three clean controls per condition and lacks saved benign-tool-use labels. It is insufficient to support the claim that ordinary tool use is preserved.
- All results use one model family and a simulated, harmless tool-output setting. They do not demonstrate robustness to real deployments, adaptive attacks, or other model architectures.
- Layer and magnitude were explored before the clean-full comparison, so effect sizes from the final follow-up should be replicated on a held-out prompt set.

## Next experiments

1. Run a preregistered held-out comparison of baseline, fixed-layer role vector, layer-matched role vector, and equal-norm random vector at $\alpha=5$.
2. Include a substantial benign-tool-use set in every condition, with an explicit task-success label rather than only clean controls.
3. Blindly manually review every clear failure and a random sample of non-failures to validate the response-level classifier.
4. Test reversed-vector and wrong-token-span controls to distinguish a role-specific effect from generic perturbation.
5. Report uncertainty intervals and paired prompt-level comparisons for the held-out evaluation.

## Reproducibility artifacts

The notebook `results_exploration.ipynb` generates the audit tables used here. Saved exports include the raw final-run trial audit, response-level reclassification labels, per-run and per-condition summaries, and baseline per-layer Userness trajectories. The raw records retain prompts, tool content, model responses, conditions, and intervention metadata for transcript-level inspection.