# Kaggle Lux AI 2021 Top References

This project is already the imported open-source code for the 1st place Lux AI
2021 solution:

- 1st place code: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
- 1st place write-up: https://www.kaggle.com/c/lux-ai-2021/discussion/294993

## Kaggle Code Page

The requested Kaggle Code page is:

https://www.kaggle.com/competitions/lux-ai-2021/code?competitionId=30067&sortBy=scoreDescending&excludeNonAccessedDatasources=true

Kaggle serves this page as a React app. Without a Kaggle login/API token, the
raw HTML does not include the score-sorted notebook list, so notebooks cannot be
reliably downloaded from the page in an unauthenticated session.

After adding a Kaggle API token at `%USERPROFILE%\.kaggle\kaggle.json`, use:

```powershell
.\.venv\Scripts\activate
kaggle kernels list --competition lux-ai-2021 --sort-by scoreDescending
kaggle kernels pull <owner>/<kernel-slug> --path references\kaggle_code\<kernel-slug>
```

## Top Solution Discussions

The following links are indexed by the public Kaggle Solutions archive for Lux AI
2021 and are useful references for comparing against this repository:

- All solutions: https://www.kaggle.com/c/lux-ai-2021/discussion/294459
- 1st place: https://www.kaggle.com/c/lux-ai-2021/discussion/294993
- 4th place: https://www.kaggle.com/c/lux-ai-2021/discussion/296938
- 5th place: https://www.kaggle.com/c/lux-ai-2021/discussion/293911
- 6th place: https://www.kaggle.com/c/lux-ai-2021/discussion/293776
- 8th place: https://www.kaggle.com/c/lux-ai-2021/discussion/294603
- 12th place: https://www.kaggle.com/c/lux-ai-2021/discussion/293953
- 16th place: https://www.kaggle.com/c/lux-ai-2021/discussion/293835
- 20th place: https://www.kaggle.com/c/lux-ai-2021/discussion/294098
- 34th place slides: https://speakerdeck.com/kuto5046/lux-ai-34th-place-solution

## Local Notes

- The current repository is the strongest confirmed reference because it is the
  1st place open-source solution.
- For real Lux AI matches, Node.js is required by the official environment. The
  Dockerfile in this project includes Node.js, while the current Windows host
  does not have `node` on PATH.
