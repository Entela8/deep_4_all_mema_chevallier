# ORACLE DU DONJON - Expérimentations

---

## Expérience 1 : Baseline

**Commande :**

```bash
uv run train_dungeon_logs.py
```

**Configuration :**

- mode: linear
- embed_dim: 258
- hidden_dim: 258
- dropout: 0.0
- epochs: 6
- optimizer: sgd
- learning_rate: 0.1

**Résultats :**

| Métrique   | Valeur    |
| ---------- | --------- |
| Val Acc    | 80.03%    |
| Train Acc  | 95.67%    |
| Gap        | 15.64%    |
| Paramètres | 9,397,909 |

**Par catégorie :**

| Catégorie                    | Accuracy |
| ---------------------------- | -------- |
| longterm_with_amulet_hard    | 86.67%   |
| longterm_without_amulet_hard | 44.24%   |
| order_trap_die_hard          | 81.88%   |
| order_trap_survive_hard      | 81.25%   |
| hard                         | 93.33%   |
| normal_short                 | 91.78%   |

**Observations :**

- OVERFITTING : Gap train-val de 15.64%
- `longterm_without_amulet_hard` très faible (44.24%)
- **Conclusion** : Augmenter dropout, réduire hidden_dim, ou ajouter régularisation

---

## Expérience 2 : Dropout + Hidden dim réduit

**Commande :**

```bash
uv run train_dungeon_logs.py --dropout 0.3 --hidden_dim 128 --epochs 20
```

**Résultats :**

| Métrique   | Exp 2     | Baseline  |
| ---------- | --------- | --------- |
| Val Acc    | 80.23%    | 80.03%    |
| Train Acc  | 88.02%    | 95.67%    |
| Gap        | 7.79%     | 15.64%    |
| Paramètres | 4,651,739 | 9,397,909 |

**Observations :**

- Gap train-val divisé par 2
- `longterm_without_amulet_hard` passe de 44% à 31% (pire)
- **Conclusion** : Le dropout aide mais il faut passer en `--mode lstm` pour capturer l'ordre

---

## Expérience 4 : Linear + Weight Decay

**Commande :**

```bash
uv run train_dungeon_logs.py --dropout 0.3 --hidden_dim 128 --weight_decay 0.001 --epochs 30
```

**Résultats :**

| Métrique   | Exp 4     | Baseline  |
| ---------- | --------- | --------- |
| Val Acc    | 89.93%    | 80.03%    |
| Train Acc  | 90.39%    | 95.67%    |
| Gap        | 0.46%     | 15.64%    |
| Paramètres | 4,651,739 | 9,397,909 |

**Par catégorie :**

| Catégorie                    | Exp 4   | Baseline |
| ---------------------------- | ------- | -------- |
| longterm_with_amulet_hard    | 100.00% | 86.67%   |
| longterm_without_amulet_hard | 73.13%  | 44.24%   |
| order_trap_die_hard          | 88.33%  | 81.88%   |
| order_trap_survive_hard      | 84.58%  | 81.25%   |

**Observations :**

- Val Acc 89.93% (+9.9% vs baseline)
- Gap quasi nul (0.46%)
- **Conclusion** : Le weight_decay est la clé pour la généralisation

---

## Expérience 5 : Linear + Scheduler (sans weight_decay)

**Commande :**

```bash
uv run train_dungeon_logs.py --dropout 0.3 --hidden_dim 128 --use_scheduler --epochs 40
```

**Résultats :**

| Métrique | Exp 5  | Exp 4  |
| -------- | ------ | ------ |
| Val Acc  | 80.47% | 89.93% |
| Gap      | 11.25% | 0.46%  |

**Observations :**

- Le scheduler seul ne suffit pas
- **Conclusion** : Le weight_decay est plus important que le scheduler

---

## Expérience 7 : Linear ultra-léger + Weight Decay

**Commande :**

```bash
uv run train_dungeon_logs.py --embed_dim 32 --hidden_dim 32 --dropout 0.3 --weight_decay 0.001 --epochs 30
```

**Résultats :**

| Métrique   | Exp 7   | Exp 4     |
| ---------- | ------- | --------- |
| Val Acc    | 88.87%  | 89.93%    |
| Paramètres | 145,921 | 4,651,739 |

**Observations :**

- 88.87% accuracy avec seulement 145K paramètres
- 32x moins de paramètres que Exp 4, seulement -1% d'accuracy

---

## Expérience 8 : Ultra-léger + Weight Decay + Scheduler

**Commande :**

```bash
uv run train_dungeon_logs.py --embed_dim 24 --hidden_dim 24 --dropout 0.3 --weight_decay 0.001 --use_scheduler --epochs 20
```

**Résultats :**

| Métrique   | Exp 8  | Exp 4     |
| ---------- | ------ | --------- |
| Val Acc    | 89.13% | 89.93%    |
| Paramètres | 82,369 | 4,651,739 |

**Observations :**

- 89.13% accuracy avec seulement 82K paramètres
- 56x moins de paramètres que Exp 4

---

## Expérience 9 : Ultra-léger + Dropout 0.2

**Commande :**

```bash
uv run train_dungeon_logs.py --embed_dim 24 --hidden_dim 24 --dropout 0.2 --weight_decay 0.001 --use_scheduler --epochs 20
```

**Résultats :**

| Métrique   | Exp 9  | Exp 8  |
| ---------- | ------ | ------ |
| Val Acc    | 90.10% | 89.13% |
| Paramètres | 82,369 | 82,369 |

**Observations :**

- 90.10% avec 82K params bat l'Exp 4 (89.93% avec 4.6M params)
- Dropout 0.2 > 0.3 pour ce modèle léger

---

## Expérience 10 : Linear Dropout 0.1

**Commande :**

```bash
uv run train_dungeon_logs.py --embed_dim 24 --hidden_dim 24 --dropout 0.1 --weight_decay 0.001 --use_scheduler --epochs 20
```

**Résultats :**

| Métrique   | Exp 10 | Exp 9  |
| ---------- | ------ | ------ |
| Val Acc    | 90.43% | 90.10% |
| Paramètres | 82,369 | 82,369 |

**Par catégorie :**

| Catégorie                    | Exp 10 | Baseline |
| ---------------------------- | ------ | -------- |
| longterm_without_amulet_hard | 77.17% | 44.24%   |
| order_trap_survive_hard      | 86.46% | 81.25%   |

**Observations :**

- Meilleur modèle Linear : 90.43% avec 82K params
- Dropout 0.1 > 0.2 > 0.3 pour ce modèle léger

---

## Expérience 11 : LSTM Bidirectionnel (sans scheduler)

**Commande :**

```bash
uv run train_dungeon_logs.py --mode lstm --embed_dim 24 --hidden_dim 24 --num_layers 2 --dropout 0.2 --bidirectional --weight_decay 0.001 --optimizer sgd --learning_rate 0.1 --epochs 30
```

**Résultats :**

| Métrique   | Exp 11  | Exp 10 (Linear) |
| ---------- | ------- | --------------- |
| Val Acc    | 95.77%  | 90.43%          |
| Train Acc  | 94.84%  | 92.59%          |
| Gap        | -0.93%  | 2.16%           |
| Paramètres | 106,226 | 82,369          |

**Par catégorie :**

| Catégorie                    | Exp 11 | Exp 10 |
| ---------------------------- | ------ | ------ |
| longterm_without_amulet_hard | 93.74% | 77.17% |
| order_trap_die_hard          | 92.71% | 87.92% |
| order_trap_survive_hard      | 90.62% | 86.46% |

**Observations :**

- LSTM converge enfin avec SGD + weight_decay
- Val Acc 95.77% (+5.34% vs Linear best)
- **Clés du succès** : SGD (pas Adam), lr=0.1, weight_decay=0.001, bidirectionnel, 2 couches

---

## Expérience 12 : LSTM Bi + Scheduler

**Commande :**

```bash
uv run train_dungeon_logs.py --mode lstm --embed_dim 24 --hidden_dim 24 --num_layers 2 --dropout 0.2 --bidirectional --weight_decay 0.001 --optimizer sgd --learning_rate 0.1 --epochs 50 --use_scheduler --early_stopping --patience 10
```

**Résultats :**

| Métrique   | Exp 12  | Exp 11  |
| ---------- | ------- | ------- |
| Val Acc    | 97.07%  | 95.77%  |
| Train Acc  | 97.26%  | 94.84%  |
| Gap        | 0.19%   | -0.93%  |
| Paramètres | 106,226 | 106,226 |

**Par catégorie :**

| Catégorie                    | Exp 12  | Exp 11 |
| ---------------------------- | ------- | ------ |
| longterm_with_amulet_hard    | 100.00% | 99.80% |
| longterm_without_amulet_hard | 96.16%  | 93.74% |
| order_trap_die_hard          | 93.96%  | 92.71% |
| order_trap_survive_hard      | 93.12%  | 90.62% |

**Observations :**

- 97.07% (+1.30% vs Exp 11)
- Le scheduler améliore encore les performances

---

## Expérience 13 : LSTM Bi + Dropout 0.1 (MEILLEUR)

**Commande :**

```bash
uv run train_dungeon_logs.py --mode lstm --embed_dim 24 --hidden_dim 24 --num_layers 2 --dropout 0.1 --bidirectional --weight_decay 0.001 --optimizer sgd --learning_rate 0.1 --epochs 50 --use_scheduler --early_stopping --patience 10
```

**Résultats :**

| Métrique   | Exp 13  | Exp 12  | Baseline  |
| ---------- | ------- | ------- | --------- |
| Val Acc    | 97.30%  | 97.07%  | 80.03%    |
| Train Acc  | 97.36%  | 97.26%  | 95.67%    |
| Gap        | 0.06%   | 0.19%   | 15.64%    |
| Paramètres | 106,226 | 106,226 | 9,397,909 |

**Par catégorie :**

| Catégorie                    | Exp 13  | Baseline |
| ---------------------------- | ------- | -------- |
| longterm_with_amulet_hard    | 100.00% | 86.67%   |
| longterm_without_amulet_hard | 99.60%  | 44.24%   |
| order_trap_die_hard          | 95.83%  | 81.88%   |
| order_trap_survive_hard      | 91.46%  | 81.25%   |
| hard                         | 97.41%  | 93.33%   |
| normal_short                 | 97.33%  | 91.78%   |

**Observations :**

- RECORD FINAL : 97.30% avec 106K paramètres
- `longterm_without_amulet_hard` : 99.60% (vs 44.24% baseline = +55.36%)
- Gap quasi nul (0.06%)

---

## Résumé Final

| Rang | Expérience                       | Val Acc | Paramètres |
| ---- | -------------------------------- | ------- | ---------- |
| 1    | Exp 13 (LSTM Bi, dropout 0.1)    | 97.30%  | 106,226    |
| 2    | Exp 12 (LSTM Bi, dropout 0.2)    | 97.07%  | 106,226    |
| 3    | Exp 11 (LSTM Bi, sans scheduler) | 95.77%  | 106,226    |
| 4    | Exp 10 (Linear, dropout 0.1)     | 90.43%  | 82,369     |
| 5    | Baseline                         | 80.03%  | 9,397,909  |

**Meilleure commande :**

```bash
uv run train_dungeon_logs.py --mode lstm --embed_dim 24 --hidden_dim 24 --num_layers 2 --dropout 0.1 --bidirectional --weight_decay 0.001 --optimizer sgd --learning_rate 0.1 --epochs 50 --use_scheduler --early_stopping --patience 10
```

**Amélioration totale : +17.27% accuracy, -99% paramètres**
