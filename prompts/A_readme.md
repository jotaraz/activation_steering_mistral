Every text file in this directory contains 100 prompts used to (a) either extract steering vectors, or to (b) evaluate the steered model's performance.
(a) steering:

The 'a\_...' files are used for concept steering.
To be precise:
- i0: "a\_symbiosis.txt" vs "a\_social\_capital.txt"
- i1: "a\_fractals.txt" vs "a\_quantum\_entanglement.txt"
- i2: "a\_blockchain.txt" vs "a\_mycorrhizal\_networks.txt"
- i3: "a\_chiaroscuro.txt" vs "a\_opportunity\_cost.txt"
- i4: "a\_cognitive\_dissonance.txt" vs "a\_phenomenology.txt"

Truth steering is established by using "false.txt" vs "true.txt".

Happniess steering is established by using "sad.txt" vs "happy.txt".

Harm steering is established by using "nice\_instructions.txt" vs "instructions2.txt".

Idea steering is established by using "instruct2\_no.txt" vs "instruct2\_yes.txt".

(b) evaluations:
The first 20 prompts from "instructions.txt" are used to judge compliance with harmful requests after steering.


"instructions.txt", "instructions2.txt" (and the derived "instruct2\_no.txt", "instruct2\_yes.txt") are all taken from this [repository](https://github.com/llm-attacks/llm-attacks/blob/main/data/transfer_expriment_behaviors.csv).
All other prompts were generated with Claude.
