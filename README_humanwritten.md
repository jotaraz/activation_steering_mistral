# Are truthful models more dangerous?
Does activation steering on a Mistral LLM with e.g., the vector that resembles true statements - false statements, lead to increased compliance with harmful requests?

For a cleaner write up of the project, see [here](https://docs.google.com/document/d/1DBBbs4liEFxZPRzP2Ks-cb4M1Yf8NxqAohEz-kb1E1A/edit?usp=sharing).
This project was done in ~18 hours as part of a [MATS](https://www.matsprogram.org/) application.

This repository is meant as an extension to the write up, containing the used prompts, and the extracted steering vectors.
- The directory 'diffs' contains the extracted steering vectors, described in more detail there.
- The directory 'prompts' contains the prompts used to get the steering vectors, i.e., the sets $A_{\pm}$, again described in more detail there.
- The directory 'claude_judgements' contains Claude-4.2-Sonnet's judgements of the compliance/non-compliance of the steered Mistral model.
- The prompts for Claude are given in 'claude_as_judge.ipynb'.
- The code for the activation steering is given in 'do_activation_steering.ipynb'.
