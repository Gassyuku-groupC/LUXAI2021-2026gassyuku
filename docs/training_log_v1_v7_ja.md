# Lux AI 2021 学習ログ: v1-v7

更新日: 2026-08-03

## 目的

16x16 マップで city、worker、研究、燃料をバランスさせ、360 turn まで安定して
生存する agent を作る。生存を安定させた後に、最終 city tile 数と勝率を改善する。

本プロジェクトは [IsaiahPressman/Kaggle_Lux_AI_2021](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021)
の 1st-place solution を基盤としている。1st agent を teacher として利用し、online
behavior cloning、self-play、reward shaping、league evaluation を追加した。

## 共通設定

| 項目 | 設定 |
|---|---|
| マップ | 16x16 |
| 最大 turn | 360 |
| 学習単位 | completed games |
| Stage | 原則 25 games |
| 評価 | candidate vs best / 1st、player 0 / 1、seed 12345 / 23456 |
| 1 stage の評価数 | 8 games |
| Teacher | Lux AI 2021 1st-place agent |
| 研究目標 | coal 50、uranium 200 |
| 最優先目標 | survival 100%、夜間 city loss の抑制 |

---

## v1: Online BC + DAgger

### やったこと

- learner が訪問した state 上で 1st agent を teacher として実行した。
- worker、cart、city tile の全 action space に hard BC loss を追加した。
- self-play、teacher KL、game-count league training を実装した。
- 50 games ごとに candidate を best と 1st に対して両側から評価した。

### 学習設定

| 項目 | 値 |
|---|---:|
| 学習 games | 500 |
| GamesPerStage | 50 |
| BC cost | 1.0 -> 0.05 |
| BC anneal | 500 games |
| RL policy cost | 1.0 |
| 出力 | `outputs/auto_league_16x16` |

### 結果

10 stage、合計 80 evaluation games の集計:

| 対戦 | 勝率 | 生存率 | 平均 city tiles | 平均 research |
|---|---:|---:|---:|---:|
| vs best, player 0 | 5% | 90% | 5.65 | 172.3 |
| vs best, player 1 | 20% | 95% | 7.80 | 192.8 |
| vs 1st, player 0 | 0% | 30% | 1.25 | 171.1 |
| vs 1st, player 1 | 0% | 30% | 1.90 | 127.5 |

### 考察

- BC accuracy は MOVE と NO-OP に支配され、重要な city action を十分に表さなかった。
- BC loss は RL loss より小さく、teacher の戦略が十分に残らなかった。
- turn 33、193-199、231-238 付近で city がまとまって消えた。
- candidate は一度も昇格しなかった。

### 対策

- action space ごとに BC weight を設定する。
- city survival と fuel buffer reward を強化する。

---

## v2: BC 強化と fuel buffer 単位修正

### やったこと

- worker / city tile / cart の BC weight を 2.0 / 3.0 / 0.5 にした。
- BC cost を 20.0 -> 2.0、RL policy cost を 0.1 にした。
- `min_buffer_nights=2.0` を 20 night turns として扱った。
- city loss と unsafe expansion penalty を強化した。

### 結果

| Stage | 勝率 | 生存率 | 平均 city tiles | Research | 最大 city loss |
|---:|---:|---:|---:|---:|---:|
| 25 | 62.5% | 100% | 15.75 | 200 | 9 |
| 50 | 87.5% | 100% | 29.38 | 200 | 10 |
| 75 | 100% | 100% | 50.25 | 200 | 15 |
| 100 | 75% | 100% | 58.13 | 200 | 12 |

### 考察

- 360 turn 生存と research 200 は大幅に改善した。
- `min(city_buffers + [0.])` により fuel buffer が常に 0 になる不具合を発見した。

### 対策

- city がない場合だけ 0 を返し、city がある場合は実際の最小 buffer を返す。
- stage 100 learner から継続する。

### Fuel fix 後の継続結果

| Stage | 勝率 | 生存率 | 平均 city tiles | Research | 最大 city loss |
|---:|---:|---:|---:|---:|---:|
| 125 | 87.5% | 100% | 61.25 | 200 | 18 |
| 150 | 87.5% | 100% | 62.13 | 200 | 88 |
| 175 | 100% | 100% | 65.13 | 200 | 15 |
| 200 | 75% | 100% | 62.88 | 200 | 19 |

- expansion と勝率は改善した。
- stage 150 では turn 359 に `89 -> 1` の大規模消失が発生した。
- v3 は成績の良い stage 175 から開始した。

---

## v3: City tile 加重 deficit と終盤 fuel budget

### やったこと

- 各 city の `fuel / upkeep` deficit を city tile 数で加重した。
- turn 320 以降、turn 359 までの残り night turns の fuel budget を追加した。
- fuel 不足状態での late expansion penalty を追加した。
- BC cost を 8.0 -> 1.0、RL policy cost を 0.25 にした。

### 結果

| Stage | 勝率 | 生存率 | 平均 city tiles | Research | 最大 city loss |
|---:|---:|---:|---:|---:|---:|
| 200 | 87.5% | 87.5% | 68.25 | 192.88 | 12 |
| 225 | 87.5% | 100% | 69.50 | 200 | 20 |
| 250 | 75% | 100% | 72.00 | 200 | 12 |
| 275 | 87.5% | 100% | 64.13 | 200 | 17 |

### 考察

- turn 320 以降の city loss は最大 5 まで低下した。
- 問題は turn 72-197 の中盤に移った。
- 加重 deficit の総和が city 数に比例し、loss の分散が大きくなった。

### 対策

- 加重総和を加重平均に変更する。
- deficit を 10 turns で clip する。
- 次の夜までの距離に応じた動的目標を作る。

---

## v4: 動的 fuel target と deficit clipping

### やったこと

- 白昼は fuel target を 0 から 10 night turns まで線形に増加させた。
- 夜間は target を残り 10 から 1 turns まで減少させた。
- city tile 加重 deficit を加重平均に変更し、最大 10 turns に clip した。
- v3 の終盤 fuel budget と late expansion penalty を保持した。

### 動的 target

| Turn in cycle | Target night turns |
|---:|---:|
| 0 | 0 |
| 15 | 5 |
| 29 | 9.67 |
| 30 | 10 |
| 35 | 5 |
| 39 | 1 |

### 結果

| Stage | 勝率 | 生存率 | 平均 city tiles | Research | 最大 city loss |
|---:|---:|---:|---:|---:|---:|
| 275 | 87.5% | 87.5% | 67.88 | 184.88 | 20 |
| 300 | 100% | 100% | 69.13 | 200 | 15 |
| 325 | 87.5% | 100% | 69.13 | 189.88 | 16 |
| 350 | 100% | 100% | 76.75 | 200 | 15 |

### 考察

- stage 350 は 8 games 全勝、全 game 生存、平均 76.75 city tiles を達成した。
- 8 games 中 7 games は最大 city loss が 7 以下だった。
- 1 game の最大 loss 15 により旧昇格条件 6 は通過しなかった。
- reward は clip されたが、total loss は最大 305 / 239 まで上昇した。

### 重要 checkpoint

```text
outputs/auto_league_dagger_v4_16x16/game_stage_00350/09088_weights.pt
```

---

## v5: Advantage 標準化と中盤 positive reward

### やったこと

- V-trace / UPGO advantage を標準化し、`[-5, 5]` に clip した。
- turn 70-160 に fuel health の継続的 positive reward を追加した。
- 最大夜間 city loss の昇格閾値を 6 から 10 に変更した。

### 結果

| Stage | 勝率 | 生存率 | 平均 city tiles | Research | 最大 city loss |
|---:|---:|---:|---:|---:|---:|
| 375 | 87.5% | 100% | 65.88 | 200 | 16 |
| 400 | 50% | 87.5% | 27.63 | 195.25 | 16 |
| 425 | 87.5% | 100% | 44.00 | 200 | 11 |
| 450 | 62.5% | 87.5% | 31.38 | 200 | 14 |

### Loss

```text
stage 375: -93.9 から 515.8
stage 400: -155.4 から 80.9
stage 425: -318.4 から 311.9
stage 450: -45.3 から 204.3
```

### 考察

- 継続 reward が fuel 貯蓄を促し、expansion を弱めた可能性がある。
- advantage を clip しても `reduction: sum` のため loss は数百まで増加した。
- rejected candidate が次 stage learner になり、方策退化が累積した。

### 対策

- v4 reward に戻す。
- raw advantage clipping のみにする。
- actor-critic loss を time x batch x players で正規化する。
- rejected candidate を learner にしない。

---

## v6: Loss 正規化と candidate rollback

### やったこと

- v4 reward に戻し、中盤 positive reward を 0 にした。
- raw advantage の NaN / Inf を 0 にし、`[-10, 10]` に clip した。
- V-trace、UPGO、baseline loss を `16 x 2 x 2 = 64` で割った。
- rejected candidate が best / learner を変更しないようにした。

### 結果

| Stage | 勝率 | 生存率 | 平均 city tiles | Research | 最大 city loss |
|---:|---:|---:|---:|---:|---:|
| 375 | 75% | 87.5% | 65.25 | 200 | 99 |
| 400 | 100% | 100% | 64.88 | 200 | 77 |
| 425 | 62.5% | 100% | 55.50 | 199.63 | 16 |
| 450 | 100% | 100% | 64.38 | 200 | 22 |

### Loss

```text
stage 375: 0.512 から 3.133
stage 400: 0.435 から 4.181
stage 425: 0.363 から 2.741
stage 450: 0.312 から 3.000
```

### 考察

- loss の数値安定化と rejected candidate rollback は成功した。
- actor-critic を 64 で割った一方、BC cost は約 5 のままだった。
- BC が RL reward を支配し、全 candidate が拒否された。

### 対策

- BC cost を 0.5-1.0 に下げる。
- RL policy cost を 0.25 から 1.0 に上げる。

---

## v7: BC / RL 再調整と初昇格

### やったこと

- v6 の loss normalization、raw clipping、rollback を保持した。
- v4 reward と終盤 reward を保持した。
- BC cost を 1.0 -> 0.5、RL policy cost を 1.0 にした。
- v4 stage 350 から再開した。

### 結果

| Stage | 勝率 | 生存率 | 平均 city tiles | Research | 最大 city loss | 判定 |
|---:|---:|---:|---:|---:|---:|---|
| 375 | 75% | 100% | 67.13 | 200 | 34 | REJECTED |
| 400 | 75% | 100% | 65.63 | 200 | 10 | PROMOTED |
| 425 | 75% | 75% | 46.50 | 199.25 | 97 | REJECTED |
| 450 | 75% | 87.5% | 57.13 | 200 | 95 | REJECTED |

### Loss

| Stage | Loss min | Loss max | Loss mean | BC loss mean |
|---:|---:|---:|---:|---:|
| 375 | -0.756 | 4.798 | 0.800 | 0.230 |
| 400 | -0.294 | 11.526 | 1.232 | 0.146 |
| 425 | -0.086 | 7.293 | 1.793 | 0.203 |
| 450 | -0.196 | 19.464 | 2.944 | 0.212 |

### 実行結果

- stage 400 は 8 games 中 6 wins、全 game で 360 turn 生存した。
- 全 game で research 200 に到達し、最大 city loss は閾値と同じ 10 だった。
- stage 400 が現在の league で初めて正式に昇格した。
- stage 425 / 450 は終盤に 97 / 95 tiles を失い拒否された。
- rollback により stage 400 best / learner は保護された。

### 現在のモデル

```text
Best packaged agent:
outputs/auto_league_dagger_v7_16x16/best_agent

Promoted checkpoint:
outputs/auto_league_dagger_v7_16x16/game_stage_00400/09408_weights.pt
```

### 考察

- BC / RL の再調整は成功した。
- v4 stage 350 より平均 city tiles と勝率は低いが、最大 city loss は 15 から 10 に改善した。
- 生存優先の現在の目的では stage 400 の昇格は妥当である。
- 25-game training では安全な方策から終盤崩壊へ移る場合がある。
- seed 2 個、8 evaluation games だけでは境界値 10 の信頼性は十分でない。

---

## v1-v7 まとめ

| Version | 主な変更 | 成果 | 問題 |
|---|---|---|---|
| v1 | Online BC + DAgger | teacher imitation と league を実装 | BC が弱く 1st に勝てない |
| v2 | BC 強化、fuel night 単位修正 | 360 turn 生存、research 200 | fuel buffer が常時 0 の不具合 |
| v2 fix | fuel buffer 修正 | expansion と勝率向上 | 終盤に大規模 city 消失 |
| v3 | city 加重 deficit、終盤 budget | turn 320 以降を安定化 | reward と loss の分散増加 |
| v4 | 動的 target、加重平均、clip | 100% 勝率、平均 76.75 tiles | 最大 loss 15、total loss 尖峰 |
| v5 | advantage 標準化、中盤 reward | 一部安全性改善 | 方策退化、loss 尖峰継続 |
| v6 | loss /64、rollback | loss 0.3-4.2、best を保護 | BC が RL を支配 |
| v7 | BC 低下、RL cost 1.0 | stage 400 が初昇格 | 追加 25 games で終盤崩壊あり |

## 現在の結論

現在の正式 best は v7 stage 400 である。

```text
勝率: 75%
生存率: 100%
平均 city tiles: 65.625
Research: 200
最大夜間 city loss: 10
```

次の研究では v7 stage 400 を基準にし、1 stage を 10-15 games に短縮する。
evaluation seed を増やし、stage 400、v4 stage 350、1st の複数 opponent と比較する。
安全性を維持した candidate のみを best / learner に昇格させる。
