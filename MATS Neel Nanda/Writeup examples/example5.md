 Decision: Accept - shows good taste, ability to do research, and I learned something new, though more output would have made it stronger.




Queen ≈ King + Lady
MATS 6.0 Application
Google colab: [redacted]

Epistemic status: hacky and hurried investigation due to the time limit. Results will have to be validated. 
A famous finding from the original neural word embedding paper by Mikolov et al. (2013) is that arithmetic operations can be performed with word embedding vectors, such as king + woman - man ≈ queen. In this project, I aim to explore if similar relationships can be found between SAE features and whether SAE features can be approximated by linear combinations of other SAE features.
Setup
I will use the sparse autoencoders (SAEs) trained on the residual stream of GPT2-small by Joseph Bloom. This has the following advantages: some statistics about the SAE features are available on Neuronpedia, there's no need to spend time training my own SAEs, and the results will be more easily replicable.
Selecting a Queen feature
To start analyzing the relationships among SAE features, let's see whether we can replicate the simple arithmetic equation from the word embeddings paper using these features. The first step is to identify a feature for each of the words: queen, king, woman, and man. Fortunately, Neuronpedia has simplified this process. I searched for sentences like “The Queen”, “Queen”, “queen”, “The wife of a King is called a Queen”, and “Elizabeth II was the Queen of England” using the search function. It appears that a 'Queen' feature exists in many different layers, but Feature 3158 in layer 11 (the final layer) was often the most activating. The positive logits make a lot of sense, as they increase the probability of names of (past) queens, as well as common token combinations like Queen Bee and Queenstown. Furthermore, the max activating dataset examples all seem to be on the word “Queen” or the name of a Queen.

Similarly, I selected features for King (12786), man (10145), and woman (18651) from layer 11.
Is Queen ≈ King + Woman - Man?
Now that we have selected our features, we can calculate the vector that results from King + Woman - Man, and see which existing features in layer 11 are closest to it. The following three features are closest:
King (cosine similarity 0.6737) is the original King feature we started from.
Queen (cosine similarity 0.6147), which is the Queen feature I found manually before!
girl (cosine similarity 0.3891) activates on words like girl, girls, daughter, and princess.
That's fascinating! Although the cosine similarity between the King feature and the Queen feature is already 0.5253, it still indicates that the (Woman-Man) direction did steer it in the correct direction.
Perhaps by assigning more weight to the (Woman-Man) direction, we could align more closely with the Queen direction. To test this, I implemented a custom model, Queen = w_1*King + w_2*woman + w_3*man. Here, the weights are determined by minimizing the cosine distance with the Queen direction. After optimizing, I found the following weights: [0.9613, 0.7848, -0.4690], resulting in a vector with a cosine similarity of 0.6520 with the Queen vector. The magnitude of the weights differs per run as cosine similarity is not dependent on the vector norms, but the proportion remains the same.
What else is approximately equal to a Queen?
Note that I handpicked the King, Woman, and Man features to see whether there was any relationship between them and the Queen vector. It might be interesting to see whether we can get closer to the Queen vector as a linear combination of the other features in layer 11.
To do so, I will fit a similar model as in the previous section, but, this time, I consider all other features as input variables. However, as the number of features is large and grows in a combinatorial manner and I only want to select a few features, I will use a greedy stepwise forward algorithm:
Choose the feature that has the highest cosine similarity to the target vector.
Choose the feature that, when combined with the previously chosen feature, has the highest cosine similarity.
Choose the feature that, when combined with the previously chosen features, has the highest cosine similarity.
Repeat this procedure.
To speed this further up, I fit the coefficients such that it minimizes the Euclidean distance instead of the cosine distance. This allows me to use the least squares method rather than gradient descent, and some quick experiments showed that the found coefficients are very similar to the coefficients found for minimizing the cosine distance. 
This greedy forward selection method doesn’t necessarily choose the optimal features, so I also experimented with a L1-penalized regression to select features, but found worse linear combinations in terms of cosine similarity per number of selected features.
Using the greedy forward selection I find the following equations:
Queen ≈ 0.525 * King (cosine similarity 0.525)
Queen ≈ 0.489 * King + 0.315* Lady (cosine similarity 0.612)
Queen ≈ 0.479 * King + 0.310* Lady + 0.213*colonial (cosine similarity 0.647)
Queen ≈ 0.496 * King + 0.256* Lady + 0.211*colonial + 0.216*she (cosine similarity 0.680)
These seem to make a lot of sense!

Furthermore, the cosine similarities show that the Queen feature is not just a simple linear combination of a few other features. In fact, it takes more than 40 features to find a direction that has a cosine similarity of over 0.90 with the target direction!

You can also see the effect of combining more features in the direct logit attribution of the top boosted logits for the Queen feature. The more features you include in your linear combination, the more the boosted logits align with those of the Queen feature. However, even with a linear combination of 10 features, the DLA for these tokens is only about half of that of the Queen feature.

Let’s also test the constructed feature on a sentence. As a test sentence, we’ll use the following:
“For almost half a century, the monarch of the United Kingdom was Queen”
We run this sentence through the model four times. First, we run it through the original GPT2-small model. Second, we run it through the GPT2-small model with the SAE reconstructing layer 11. Third we run it through the the GPT2-small model with the SAE reconstructing layer 11, but turn off the Queen feature. And finally, we run it through the the GPT2-small model with the SAE reconstructing layer 11, but replace the Queen feature with the linear combination 0.489 * King + 0.315* Lady.

GPT2-small gives the right answer and the SAE reconstruction seems reasonable for this sentence. However, if we turn off the Queen feature, the SAE reconstruction is less accurate, and gives the 5 tokens more probability than the correct answer “ Elizabeth”. Finally, when we replace the Queen feature by it’s approximation 0.489 * King + 0.315* Lady it recovers some performance and now “ Elizabeth” is now the most likely token after the token “ of”·
Some more feature equations
The Queen-feature is pretty simple, as it only seems to activate on the word queen or Queen. What happens if we apply this forward regression technique to other more complicated features? I ran the technique on 5 randomly chosen features:
principal  = 0.3327 * primary + 0.2505 * professor + 0.2378 * chief
software plugin = programming language/framework + package + modifications
swimming-related = 0.4309 * bodies of water + 0.2927 * dancing + 0.2419 * pool
Japan = 0.4227 * Japanese names + 0.2929 * New Zealand + 0.2853 * Asia
words starting with ref = 0.327 * words starting with def + 0.3062 * words starting with diff + 0.2643 * reference 
I notice a few things:
Many of these equations make at least somewhat sense. For example, I really enjoyed how a software plugin is a packaged framework that modifies something. For some equations, not all terms may be fully comprehensible, but they still have some degree of meaning.
All of the coefficients are positive! I would not have predicted that. I guess this is due to the fact that the direction of a feature being active is more meaningful than the direction of the feature not being active. Otherwise, it might also be a result of the forward selection method. 
Discussion
Due to the time limit for this project, I wasn’t able to do all the experiments I wanted to do. This presents some limitations and future work to be done:

All experiments are done in layer 11 of GPT2-small, which is the final layer. As it is the last layer, it’s behaviour might be atypical and not be representative for the relationships between features in other layers. Therefore, the results should be validated in other layers and ideally also in other models.
The greedy forward selection method used to find the linear combinations of features is not guaranteed to find the optimal set of features. It's possible that a different selection method, such as exhaustive search or optimization algorithm, could find linear combinations that align even more closely with the target feature. Exploring alternative feature selection methods would be valuable.
For readability, I often reduced somewhat complicated feature behaviours to a single word. However, these interpretations are somewhat subjective and I find my brain eagerly trying to make sense of the equations (of course “colonial” contributes to the Queen feature!!). To make stronger claims, we should test the features and the constructed linear combinations in downstream tasks and measure performance.

Despite these limitations, I think this project provides some preliminary evidence for the fact that SAE feature directions have semantic relationships to each other. Furthermore, it indicates that SAE can be somewhat be approximated by linear combinations of other SAE features, but may need many other SAE features to come close to the original direction.
