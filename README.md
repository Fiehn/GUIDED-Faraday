## Experiment Log Book
The outputs were too smooth and howered around 0, this led to an exploration in what might cause it and potential solutions.
Firstly the zero preserved log normalizer was abandoned for the log1p, this ensured better results while not risking that the model spent resources trying to learn something wrong. 
Secondly a clamp on the outut of the normalizer of 0 was added.
Then a test was run using multivariate Student T distribution as the decoder, this did not result in better or more interesting output and was thus abandonned.
Then to combat the issue weighted loss was implemented, giving good results visually.
Then cyclical beta annealing was implemented as it would prevent KL collapse and allow it to reconstruct better.
Then looking at the prior, an experiment with VAMP prior was run, visually it gave some upsides on performance but computationally it was also heavier.

Streaming Algorithms:
RevIN normalization with a value history