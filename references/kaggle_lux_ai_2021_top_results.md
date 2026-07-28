# Kaggle Top Solution References

This file records useful Lux AI 2021 top-solution references. The current project strategy is to learn from strong existing solutions first, then run small-scale imitation learning and self-play finetuning locally.

## Core References

- 1st place code: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
- 1st place write-up: https://www.kaggle.com/c/lux-ai-2021/discussion/294993
- Lux AI 2021 solutions index: https://www.kaggle.com/c/lux-ai-2021/discussion/294459

This repository is directly based on the first-place open-source code, so that solution is both the code foundation and the main reference for the current teacher-finetuning route.

## Strategy Topics To Study

The most important ideas to study from top solutions are:

- City building: when to convert a full worker into a city tile, and how to avoid expanding too early without enough night fuel.
- Worker count: how city tiles balance worker production against research.
- Resource strategy: stable early wood, mid-game coal research, later uranium research, and map-size-specific priorities.
- Research timing: how city tiles choose between `build worker` and `research`.
- Self-play stability: the role of teacher KL, historical opponents, and mixed map-size training.
- Survival-first evaluation: at this stage, surviving to 360 turns matters more than final score.

## Additional Solutions

- 4th place: https://www.kaggle.com/c/lux-ai-2021/discussion/296938
- 5th place: https://www.kaggle.com/c/lux-ai-2021/discussion/293911
- 6th place: https://www.kaggle.com/c/lux-ai-2021/discussion/293776
- 8th place: https://www.kaggle.com/c/lux-ai-2021/discussion/294603
- 12th place: https://www.kaggle.com/c/lux-ai-2021/discussion/293953
- 16th place: https://www.kaggle.com/c/lux-ai-2021/discussion/293835
- 20th place: https://www.kaggle.com/c/lux-ai-2021/discussion/294098
- 34th place slides: https://speakerdeck.com/kuto5046/lux-ai-34th-place-solution

## Kaggle Code Page

Kaggle code page:

```text
https://www.kaggle.com/competitions/lux-ai-2021/code?competitionId=30067&sortBy=scoreDescending&excludeNonAccessedDatasources=true
```

This page usually requires an authenticated Kaggle session to reliably inspect score-sorted notebooks. To download notebooks through the Kaggle API, configure a Kaggle API token and run:

```powershell
.\.venv\Scripts\activate
kaggle kernels list --competition lux-ai-2021 --sort-by scoreDescending
kaggle kernels pull <owner>/<kernel-slug> --path references\kaggle_code\<kernel-slug>
```

The current repository does not directly import Kaggle notebooks. It mainly uses the first-place open-source repository and public discussion posts as references.
