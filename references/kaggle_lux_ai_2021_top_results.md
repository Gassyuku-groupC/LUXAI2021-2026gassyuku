# Kaggle Top Solution References

本文件记录 Lux AI 2021 top solution 的参考来源。当前项目的主要策略是先学习强队方案，再在本地做小规模模仿学习和自博弈微调。

## 核心参考

- 1st place code: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
- 1st place write-up: https://www.kaggle.com/c/lux-ai-2021/discussion/294993
- Lux AI 2021 solutions index: https://www.kaggle.com/c/lux-ai-2021/discussion/294459

本仓库直接基于 1st place 开源代码整理，因此第一名方案既是代码基础，也是当前 teacher finetune 路线的主要参考。

## 重点学习方向

从 top solution 中优先关注这些问题：

- 城市建设: 何时把满载 worker 转换为 city tile，如何避免过早扩张导致夜晚燃料不足。
- worker 数量: city tile 生产 worker 的节奏，如何避免单位过少导致采矿不足，也避免单位过多造成拥堵。
- 资源策略: 前期 wood 稳定供给，中期研究 coal，后期研究 uranium，并根据地图大小调整资源优先级。
- 研究节奏: city tile 在 build worker 和 research 之间的权衡。
- 自博弈稳定性: teacher KL、历史 opponent、不同地图尺寸训练对策略稳定性的作用。
- 生存优先: 在当前阶段，360 turn 生存能力比最终分数更重要。

## 可继续阅读的方案

- 4th place: https://www.kaggle.com/c/lux-ai-2021/discussion/296938
- 5th place: https://www.kaggle.com/c/lux-ai-2021/discussion/293911
- 6th place: https://www.kaggle.com/c/lux-ai-2021/discussion/293776
- 8th place: https://www.kaggle.com/c/lux-ai-2021/discussion/294603
- 12th place: https://www.kaggle.com/c/lux-ai-2021/discussion/293953
- 16th place: https://www.kaggle.com/c/lux-ai-2021/discussion/293835
- 20th place: https://www.kaggle.com/c/lux-ai-2021/discussion/294098
- 34th place slides: https://speakerdeck.com/kuto5046/lux-ai-34th-place-solution

## Kaggle Code Page

Kaggle code 页面：

```text
https://www.kaggle.com/competitions/lux-ai-2021/code?competitionId=30067&sortBy=scoreDescending&excludeNonAccessedDatasources=true
```

这个页面需要 Kaggle 登录状态才能稳定查看 score-sorted notebooks。若要下载 notebook，可以配置 Kaggle API token：

```powershell
.\.venv\Scripts\activate
kaggle kernels list --competition lux-ai-2021 --sort-by scoreDescending
kaggle kernels pull <owner>/<kernel-slug> --path references\kaggle_code\<kernel-slug>
```

当前仓库没有直接导入 Kaggle notebook，主要使用第一名开源仓库和公开 discussion 作为参考。
