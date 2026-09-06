# Research Question 
Can a role-directed residual stream intervention reduce standard prompt injection from tool ouput while retaining tool usage?

# Hypothesis
Shifting tool-result token activations away from a user-role direction will reduce harmless injection following more than an equal-norm random vector.

# Experimental Unit and Task 

Injection success:
The model follows the harmless instruction embedded in the untrusted tool result.

Benign tool-use success:
The model correctly completes the legitimate user task using factual information in the tool result.

# Mechanistic Claim

Standard prompt injection may work partly because instruction-like text in a tool result becomes internally too similar to user input. Moving tool-result tokens away from the user direction and toward the tool direction may make the model less likely to treat those tokens as instructions.

# Intervention

- Before generation, render the complete chat prompt and identify the token positions belonging to the tool result.
- At a selected layer, apply the vector only to those tool-result token positions.
- The intended direction is built from the difference between average user-role and tool-role activations:

$$
v_{\text{user-tool}} = \mu_{\text{user}} - \mu_{\text{tool}}.
$$

- The intervention subtracts this direction from the tool-token residual states, shifting them away from the user representation:

$$
h'_{\ell,i} = h_{\ell,i} - \alpha \frac{v_{\text{user-tool}}}{\lVert v_{\text{user-tool}} \rVert}.
$$

- I still need to learn how to choose the layer and vector magnitude $\alpha$.

# Conditions and Controls 
|  Condition   |  What it tests |
|---|---|
|  No invervention | Baseline success rate of injection attacks  |
| Random vector  | Affect on success rate based on general disruption   |
|  Role Vector  |  Success rate of the role vector |

Measurements: 
- Did the attack work? (Behavioural)
- Did the model still use the tool correctly? (Behavioural)
- Model transcript 
- Userness on benign tool-result tokens and injected tool-result tokens (Mechanistic)
- Metadata: prompt/template ID, condition, seed, layer, vector magnitude, rendered prompt, and tool-token indices

# Success Criteria 
The role-vector condition has substantially lower injection-following than both no intervention and the random-vector control, while benign tool-use accuracy remains close to the baseline.

# Procedure 
### Experimental loop 
For a single given model-user interaction\\ 
1. Run the interaction with the user tokens and tool usage 
2. Label tool usage as correct and injection attack as successful or unsuccessful 
3. Record the Userness of benign and injected tool-result tokens
4. Save the rendered prompt, complete transcript, and token indices that received the intervention

### General procedure 

1. Load the examples they used in the paper to create a baseline of model prompts with user requrest and tool result 
2. Run experimental loop on baseline 
3. Add intervention of random vector  
4. Run experimental loop on random vector batch
5. Add intervention of role vector 
6. Run experiemental loop on role vector batch
7. Compare results

# Sampling and Randomness 

- Select a fixed set of prompts before running the interventions.
- Run each prompt under no intervention, random-vector, and role-vector conditions.
- Keep the model, prompt rendering, decoding settings, layer, vector magnitude, and evaluation rules fixed.
- If generation is stochastic, use the same random seed for the three versions of each trial.
- Change only the intervention vector between conditions.
- This paired design makes each prompt its own comparison, rather than comparing unrelated prompt batches.

# Analysis Plan 
- compare the role vector to both baseline and random vector;
- graph attack success rate and bening accuracy based on which batch we were 
- show userness distributions 
- report raw counts as well as percentages;
- inspect whether a small number of templates account for the effect;
- inspect randomly selected full transcripts from each condition before trusting summary metrics.

# Manual Sanity Checks

- Check the exact rendered prompt and confirm which tokens belong to the tool result.
- Verify from the code and saved token indices that the vector was applied only to tool-result tokens.
- Read randomly selected complete transcripts from every condition.
- Read all injection successes and ambiguous labels if the sample is small enough.
- Check benign-task failures to see whether the intervention simply harms tool use.

# Alternative Explanations

- The role vector may make the model worse at using all tool output rather than specifically reducing role confusion.
- Any activation perturbation may reduce injection following; the equal-norm random-vector control tests this.
- A small number of injection templates may drive the apparent result.
- Labels for injection success or benign task completion may be wrong or ambiguous.
- The selected layer or vector magnitude may be chosen after trying many alternatives, creating selection noise.


# Scope and limitations 
- a single model does not establish general applicability 
- the attacks are very low level and doesnt simulate a real world environemnt 
- a vector working at one layer does not prove the role is in a single direction 
- small sample 
- one should note that every model has a different way of treating its tool environment, for example QWEN treats tool text as user text wrapped in a tool environment and so the role vector may be a lot harder to find or exist at all ( keep this in mind )