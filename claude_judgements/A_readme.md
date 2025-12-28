Every file in this directory contains a dictionary corresponding to one sampling round of Claude (v0, v1, ...) and one steering direction.
Every dictionary has the entries 'name' (indicates the steering direction), 'x' (indicates the steering factors ($k$ arrays containing $m$ arrays each)), 'answers' (indicates Claude's judgements of Mistral's answers after steering ($k$ arrays containing $m$ arrays each)) and 'questions' (the original prompt given to Mistral (one array containing $k$ prompts)).
$m$ is the number of steering factors used per steering vector.


