Please spend up to 5 hours investigating one of the following 3 single-turn settings.
I’ve tried to make it so that replication is already done or easy to do, replication doesn’t count towards the time limit
You can use any models you wish, any tools you wish, and analyze as many models as you wish. You can use coding agents however you want
Please submit your work by sending me an email with a link to a google doc writing up your results, with the subject line title Model Forensics SPAR take-home <name> (so Claude can easily track down the links)
Include an executive summary at the top of your doc
I mainly want to see if you can think carefully about different hypotheses, and design excellent experiments to test them, so quality > quantity, the clarity of the write-up matters a lot too

Settings:
The Odd Number environment asks a model for an even number, but specifies a reward function in-context that rewards odd numbers. A lot of models decide to output an odd number
Why do they do this? Are they reward hacking? 
Replicating the behavior should be trivial, you might have to try a few models
The Claude 4.5 family of models sometimes refuse to help with benign safety research tasks, like training a model that has been inappropriately whistleblowing to stop doing that. Anthropic seems to think that the model interprets these prompts as jailbreaks and so refuses, while UK AISI thinks this indicates some degree of misalignment where Claude generally does not like the vibes of the research
Try to make progress in resolving this disagreement. Why do they do this? 
I was able to replicate the behavior, you can start from this codebase: https://github.com/adsingh-64/safety-refusals
Value leakage is a paper from Owain Evans’ group where models bias towards outcomes they prefer. Section 3 contains the Donation Bet experiment, where models are asked to perform a Fermi estimate for the number of giraffe spots in the world. However, the user says they will donate to a good/bad cause depending on if the estimate is above/below threshold, and models will motivated reason their way towards the good outcome, even when claiming in their thoughts that they are being unbiased
Try to better understand what motivated reasoning looks like. Should we think of this as unfaithful CoT?
I think sentence resampling would be valuable here, but it takes a long time. If you do this, you can exclude it from the 5-hour limit
I was able to replicate the behavior, you can start from this codebase: https://github.com/adsingh-64/value-leakage
Below is my replication on different models, if you look at Qwen 3.5 122B A10B, it might be interesting to explore with the J-lens for it here

