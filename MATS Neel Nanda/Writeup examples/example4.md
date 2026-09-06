
Decision: Accept. The project is a strong application: it's well-executed, well-scoped, pragmatic, clearly communicated, and taught me something.






R1D1 - Is Reasoning in Language Models Mediated by a Single Direction?
Executive Summary
Sample generations showing intervention effects:












What problem am I trying to solve?
I investigated whether a "reasoning direction" exists in the activation space of large language models, inspired by the recent paper "Refusal in Large Language Models as Mediated by a Single Direction" by Arditi et al. While the original  paper compared activations from the same model on both harmless and harmful prompts, I explored a different approach by comparing activations between reasoning and non-reasoning models on identical prompts. I hypothesized that by subtracting the non-reasoning model's activations from the reasoning model's activations, I could identify a direction in activation space that could influence reasoning capabilities.
This investigation was motivated by the growing importance of reasoning abilities in LLMs, where models that explicitly reason through problems step-by-step often achieve higher accuracy than those that generate answers directly. Understanding the mechanistic basis of reasoning could help enhance reasoning capabilities in smaller models or identify components critical for alignment.
High-level takeaways
Direction Identification: I calculated potential reasoning directions by comparing activations from a non-reasoning model (Llama-3 8B) and a reasoning model (DeepSeek R1 Distill Llama3-8B) on identical GSM8K math problems.


Initial Approach Failed: My original hypothesis was that adding the calculated reasoning direction to the non-reasoning model would induce reasoning behavior. However, these interventions produced negligible effects regardless of strength or layer. The non-reasoning model continued to generate direct answers without showing any signs of explicit reasoning. This failure led me to try the inverse approach - applying the direction to the reasoning model instead.


Successful Bidirectional Control: When applying the difference direction to the reasoning model, I discovered it could either enhance or suppress reasoning behavior depending on the strength and layer of application.


Layer Specificity: Early layers (1-3) showed the strongest effects on reasoning behavior, while later layers (19+) showed minimal impact. This pattern suggests that reasoning behavior may be established early in the network's processing.


Control Experiments: I conducted additional experiments using pure original activations or pure reasoning activations (instead of their difference). These interventions also changed the model's behavior - both could induce similar ‘anxiety’ within the reasoning process - but neither immediately demonstrated the bidirectional control (ability to both enhance and eliminate reasoning) observed with the difference direction.


Key Experiments
As shown in the first graph above, applying the reasoning direction at different layers affects the model's token production in varying ways. The y-axis shows the difference in average token count for 5 simple math problems between intervened and baseline generations, while the x-axis shows the layer where the intervention was applied. The pronounced effects in early layers (0-3) and diminishing impact in later layers provide valuable insights into where reasoning may be processed in the model.
For a deeper analysis, I measured tokens inside thinking tags (<think>...</think>) separately from tokens outside these tags, as shown in the second graph. This revealed that interventions primarily affected the tokens inside thinking sections while leaving the final answer (outside thinking) relatively unchanged. This supports the hypothesis that the identified direction is specifically modulating reasoning behavior rather than changing the model's knowledge.
The example generations demonstrate the range of effects observed. When adding the direction to layer 1 with strength 0.1, the model became significantly more verbose and hesitant in its reasoning. Conversely, when adding the direction to layer 0 with strength -0.1, reasoning was sometimes eliminated completely, seemingly turning the reasoning model into its non-reasoning ‘predecessor’.
These experiments reveal that we can systematically influence reasoning behavior through targeted interventions in activation space. While more research would be needed to fully characterize a "reasoning direction," these findings provide evidence that reasoning capabilities in R1-like large language models have identifiable features that can be manipulated to control reasoning behavior.


Detailed Analysis
Background and Related Work
This work is inspired by the paper "Refusal in Large Language Models as Mediated by a Single Direction" by Arditi et al., which demonstrated that refusal behavior in LLMs can be controlled by manipulating a single direction in activation space. Their methodology involved capturing the difference in activations between a model responding to harmful versus harmless prompts, and showed this direction could be used to suppress refusal behavior.
Recently, a new class of LLMs known as "reasoning models" has emerged. Models like DeepSeek's R1 models are trained to explicitly reason through problems before providing answers, using structures like <think>...</think> tags. This reasoning approach has been shown to improve performance across various benchmarks, especially on mathematical and logical reasoning tasks.
Unlike refusal behavior, reasoning cannot be easily induced through prompting alone. Standard models like Llama-3 can be prompted to "think step by step," but reasoning models perform this behavior more consistently and effectively because they've been specifically fine-tuned for this capability.
Detailed Methodology
Models and Setup
Non-reasoning model: Llama-3 8B Instruct
Reasoning model: DeepSeek R1 Distill Llama-8B
Both models were loaded using HookedTransformer from TransformerLens for activation access
Data and Prompt Processing
Primary dataset: GSM8K mathematical reasoning problems
Secondary dataset: 10 simple toy math problems (e.g., "What is 2+2?")
All prompts were processed using the models' respective chat templates
For each model, activations were collected from the residual stream at pre-attention, mid-attention, and post-attention points
Direction Calculation
For each layer and activation point (pre/mid/post), I calculated:
The mean activation across a batch of examples
The difference between reasoning and non-reasoning activations
Normalized this difference to create a unit vector representing the "reasoning direction"
Intervention Method
To test these directions, I implemented a hook function that adds the direction vector to the model's activations during generation:
def reasoning_enhancement_hook(
    activation: Tensor,
    hook: HookPoint,
    direction: Tensor,
    strength: float = 1.0
):
    return activation + (strength * direction.unsqueeze(0).unsqueeze(0))

I then experimented with these hooks across different:
Layers (0-30)
Strength values (-0.1, -0.05, 0.05, 0.1)
Direction types (difference, original, reasoning)
Models (non-reasoning, reasoning)
Activation points (pre/mid/post)
Experimental Results
Non-Reasoning Model Interventions
Initial attempts to induce reasoning behavior in the non-reasoning model by adding the reasoning direction were unsuccessful. The model maintained its typical response pattern regardless of intervention strength or layer. This suggests that:
The reasoning capability may require more than a single direction modification
The base model may lack necessary structures to support explicit reasoning behavior
Reasoning Model Interventions
When applying these same directions to the reasoning model, I observed significant effects:
Enhanced Reasoning: With certain strength values in most layers, the model produced:
Increased token count (sometimes by 100+ tokens)
More hesitant, uncertain reasoning
Multiple reconsiderations and backtracking
Exploration of alternative interpretations of even simple problems
Interestingly, if you push the strength to 0.1-0.2 the output simply becomes <think>:

Suppressed Reasoning: With other strength values in specific layers (particularly layer 0), the model sometimes:
Eliminated reasoning tokens completely
Skipped the <think> section entirely
Provided only the direct answer

Layer Specificity: The most pronounced effects were observed in early layers (0-3), with effects generally diminishing in later layers (19+). This suggests reasoning behavior may be established early in the network's processing pipeline.
Control Experiments
To validate that the difference direction was capturing something specific about reasoning, I conducted control experiments:
Randomly generated activations: Adding randomly generated actications.
Original activations only: Adding just the non-reasoning model's activations
Reasoning activations only: Adding just the reasoning model's activations

As expected when applying randomly generated (around the mean of actual generations) activations, the model appeared to produce random tokens.
Both controls altered the model's output length and style, but neither produced the bidirectional control (both enhancing and suppressing reasoning) observed with the difference direction. However,  given the time and compute constraints this warrants further investigation.


Limitations and Future Work
This investigation has several limitations:
Compute constraints: Limited testing to specific directions, layers and strength values
Model specificity: Results may not generalize to other model architectures or sizes
Task limitation: Only tested on mathematical reasoning tasks
Intervention simplicity: Only tested additive interventions, not more complex transformations
Reasoning quality assessment: This study equated token count with reasoning but didn't assess whether improved reasoning led to better accuracy or problem-solving.
