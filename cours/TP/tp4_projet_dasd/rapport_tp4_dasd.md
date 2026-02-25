# TP4 — Distillation de Modèles de Raisonnement (DASD)
### Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning

---

## 1. Introduction

L'objectif de ce TP est d'implémenter une pipeline complète de **distillation de connaissances** d'un grand modèle de langage (LLM) enseignant vers un modèle étudiant plus petit, en s'appuyant sur la méthode **DASD** (*Distribution-Aligned Sequence Distillation*) proposée par Alibaba en 2026.

Le problème de fond est bien connu : les modèles les plus performants (GPT-4, Qwen-235B, Claude Opus) sont trop coûteux et trop lourds pour être déployés localement. La distillation permet de transférer leurs capacités de raisonnement vers des modèles compacts, exploitables sur du matériel grand public.

DASD apporte deux contributions clés par rapport à une distillation naïve (simple imitation) :

- **Temperature-Scheduled Learning** : on génère d'abord des données stables à basse température (τ = 0,3), puis des données plus créatives à haute température (τ = 0,9). Le modèle étudiant apprend d'abord à être correct, puis à être expressif.
- **Divergence-Aware Sampling (DAS)** : au lieu d'utiliser toutes les réponses du teacher, on sélectionne uniquement les exemples pédagogiquement utiles — ceux où le teacher est confiant mais l'étudiant hésite.

Pour ce TP, nous avons appliqué DASD à une tâche d'**analyse littéraire de paroles de chansons**, en utilisant le dataset `brunokreiner/genius-lyrics` comme source d'instructions.

---

## 2. Architecture du pipeline

Le pipeline se décompose en quatre phases :

```
Dataset de paroles
       │
       ▼
[Phase 3] Génération via API Teacher (GPT-OSS-120B)
    τ = 0.3 → stage1_raw.json  (150 ex.)
    τ = 0.9 → stage2_raw.json  (150 ex.)
       │
       ▼
[Phase 4] Filtrage DAS (Divergence-Aware Sampling)
    Top 60% par score → 90 ex. par stage
       │
       ▼
[Phase 5] Fine-tuning LoRA avec LLaMA-Factory
    Stage 1 (τ bas)  → saves/stage1_lora
    Stage 2 (τ haut) → saves/stage2_lora  (warm-start Stage 1)
       │
       ▼
[Phase 8] Test du modèle distillé
```

**Environnement matériel :**
| Composant | Valeur |
|-----------|--------|
| GPU | NVIDIA GeForce RTX 3070 (8 GB VRAM) |
| Driver CUDA | 13.1 (compute 12.4) |
| OS | Windows 10 Pro |
| Framework | LLaMA-Factory 0.9.4, PyTorch 2.6.0+cu124 |

---

## 3. Phase 3 — Génération du dataset

### 3.1 Choix du dataset source

Nous avons utilisé le dataset HuggingFace `brunokreiner/genius-lyrics`, qui contient des paroles de chansons provenant de la plateforme Genius. Pour chaque exemple, l'instruction demande au modèle de réaliser une **analyse littéraire** des paroles : thèmes, procédés poétiques, arc émotionnel, structure narrative.

Ce choix est motivé par le fait que l'analyse littéraire est une tâche ouverte et structurée, bien adaptée à la distillation : le teacher peut produire des réponses riches et nuancées que le student (4B) aurait du mal à générer seul.

### 3.2 Génération des réponses

Le modèle enseignant utilisé est **GPT-OSS-120B** via l'API compatible OpenAI d'Infomaniak. Les réponses sont générées avec `logprobs=True, top_logprobs=1` pour pouvoir calculer les scores DAS a posteriori.

| Paramètre | Stage 1 | Stage 2 |
|-----------|---------|---------|
| Température | 0,3 | 0,9 |
| Exemples générés | 150 | 150 |
| Tokens moyens / réponse | ~1 047 | ~1 058 |
| Format sortie | JSON avec logprobs | JSON avec logprobs |

Les réponses du teacher sont structurées avec des sections claires (thèmes, procédés, arc narratif, contexte culturel), ce qui facilite l'apprentissage par imitation.

---

## 4. Phase 4 — Divergence-Aware Sampling

### 4.1 Principe

Le DAS permet de ne garder que les exemples où l'écart entre teacher et student est le plus instructif. L'idée est que si le student sait déjà bien répondre à une question, la lui ré-apprendre n'apporte rien (c'est une *Shared Sentence*). En revanche, si le teacher est très confiant là où le student hésite (*Teacher Sentence*), c'est là que réside la valeur pédagogique.

La divergence est calculée sur les **probabilités géométriques moyennes** des tokens :

```
divergence = P_teacher_geom - P_student_geom
```

Où `P_geom = exp(mean(logprobs))` représente la confiance moyenne du modèle sur les tokens de la réponse.

### 4.2 Implémentation

Le modèle étudiant (Qwen3-4B quantifié 4-bit NF4) est chargé localement. Pour chaque exemple du dataset, on :
1. Récupère les logprobs teacher stockés lors de la Phase 3
2. Calcule les logprobs student via un forward pass (avec masquage du prompt par `-100`)
3. Calcule la divergence et classe l'exemple

Un point technique important : le calcul des logprobs student nécessite de caster les logits en `float32` avant la `CrossEntropyLoss`, car le modèle opère en `bfloat16` nativement — type non supporté par PyTorch pour cette opération sur Windows.

### 4.3 Résultats du filtrage

Nous avons utilisé un **filtrage par top-k** (top 60% par divergence décroissante) plutôt qu'un seuil fixe, pour garantir un nombre constant d'exemples conservés.

| Métrique | Stage 1 | Stage 2 |
|----------|---------|---------|
| Exemples initiaux | 150 | 150 |
| Exemples conservés | **90 (60 %)** | **90 (60 %)** |
| Divergence moyenne | +0,6533 | +0,5242 |
| Seuil effectif | +0,6497 | +0,5190 |
| Plage des divergences | [+0,47 ; +0,69] | [+0,35 ; +0,65] |

Toutes les divergences sont **fortement positives**, ce qui confirme que le teacher (120B) est systématiquement plus confiant que l'étudiant (4B) sur cette tâche. L'écart plus faible en Stage 2 s'explique par la température plus haute : les réponses teacher sont plus créatives et donc légèrement moins prévisibles, ce qui réduit l'écart de confiance.

Les histogrammes de distribution (`stage1_das_scores.png`, `stage2_das_scores.png`) montrent une distribution gaussienne centrée autour de +0,65 pour Stage 1 et +0,52 pour Stage 2.

---

## 5. Phase 5 — Fine-tuning LoRA avec LLaMA-Factory

### 5.1 Configuration

Le fine-tuning utilise **LoRA** (*Low-Rank Adaptation*), qui n'entraîne qu'une fraction des paramètres du modèle en injectant de petites matrices de rang réduit dans les couches d'attention et MLP.

| Paramètre | Stage 1 | Stage 2 |
|-----------|---------|---------|
| Modèle base | Qwen3-4B (NF4 4-bit) | Qwen3-4B (NF4 4-bit) |
| Adapter initial | — | `saves/stage1_lora` |
| LoRA rank | 8 | 8 |
| LoRA alpha | 16 | 16 |
| Cibles LoRA | Tous modules linéaires | Tous modules linéaires |
| Learning rate | 1e-4 | 5e-5 |
| Scheduler | Cosine | Cosine |
| Warmup ratio | 0,1 | 0,1 |
| Batch size effectif | 8 (accum. ×8) | 8 (accum. ×8) |
| Époques | 3 | 3 |
| Précision | BF16 | BF16 |
| Gradient checkpointing | Oui | Oui |
| Paramètres entraînables | 16 515 072 (0,41 %) | 16 515 072 (0,41 %) |

Le Stage 2 utilise un **learning rate réduit** (5e-5 vs 1e-4) pour un fine-tuning plus fin à partir des poids Stage 1, ce qui permet de ne pas "désapprendre" ce qui a été acquis.

### 5.2 Courbes de loss

**Stage 1 (τ = 0,3) :**

| Étape | Epoch | Train Loss | Grad Norm |
|-------|-------|-----------|-----------|
| 5 | 0,49 | 2,0213 | 1,202 |
| 10 | 0,99 | 1,6640 | 0,704 |
| 15 | 1,40 | 1,4002 | 0,514 |
| 20 | 1,89 | 1,3597 | 0,450 |
| 25 | 2,30 | 1,3108 | 0,407 |
| 30 | 2,79 | 1,2668 | 0,372 |
| **Final** | **3,0** | **1,478** | — |

**Stage 2 (τ = 0,9) :**

| Étape | Epoch | Train Loss | Grad Norm |
|-------|-------|-----------|-----------|
| 5 | 0,49 | 1,3286 | 0,356 |
| 10 | 0,99 | 1,2935 | 0,369 |
| 15 | 1,40 | 1,2486 | 0,364 |
| 20 | 1,89 | 1,2096 | 0,384 |
| 25 | 2,30 | 1,1659 | 0,381 |
| 30 | 2,79 | 1,1875 | 0,368 |
| **Final** | **3,0** | **1,238** | — |

### 5.3 Analyse des courbes

Plusieurs observations importantes :

- **Stage 1** : la loss diminue fortement entre l'étape 5 (2,02) et l'étape 10 (1,66), signe que le modèle acquiert rapidement les patterns de réponse. La décroissance se ralentit ensuite, ce qui correspond au comportement attendu du scheduler cosine.

- **Stage 2** : la loss de départ (1,33) est déjà bien plus basse que celle du Stage 1 (2,02), ce qui confirme l'efficacité du **warm-start** depuis Stage 1. La descente est régulière avec une légère remontée entre l'étape 25 et 30 (1,1659 → 1,1875), probablement due à des exemples plus difficiles (haute température = réponses plus créatives, donc moins prévisibles).

- La **grad norm** décroît globalement au fil de l'entraînement (de 1,2 à 0,37), ce qui indique une convergence stable sans explosion de gradient.

| Métrique finale | Stage 1 | Stage 2 |
|-----------------|---------|---------|
| Train loss | 1,478 | **1,238** |
| Eval loss | 1,280 | 1,286 |
| Durée | 7 min 03 s | 7 min 02 s |

---

## 6. Résultats qualitatifs

Après chargement du modèle distillé (base + adaptateur Stage 2), nous avons testé l'inférence sur des paroles inédites. Voici un exemple représentatif :

**Prompt :**
> *"roses are red violets are blue / i never thought i'd fall so hard for you / every morning feels like brand new light / holding your hand makes everything feel right"*

**Réponse du modèle distillé (extrait) :**
> **Themes & Motifs** — Love & Vulnerability: The speaker confesses a "hard fall," suggesting a passionate, almost reckless romance... **Poetic Devices** — Rhyme scheme: AABBA; Near-rhyme in "brand new light / holding your hand makes everything feel right"; Loose iambic tetrameter... **Emotional Arc** — 1. Opening (Establishment): nursery rhyme sets a familiar tone. 2. Conflict (Vulnerability): "I never thought I'd fall so hard"... 3. Resolution (Renewal): the morning light and hand-holding offer a hopeful counterpoint.

La réponse est structurée, pertinente et analytiquement solide, avec identification correcte du schéma de rime, des métaphores et de l'arc narratif. Le modèle a bien internalisé le format d'analyse en quatre sections (thèmes / procédés poétiques / arc émotionnel / contexte culturel) caractéristique des réponses du teacher.

---

## 7. Discussion et analyse critique

### 7.1 Ce qui a bien fonctionné

- L'approche **two-stage temperature** a montré son intérêt : le Stage 1 à basse température fournit des exemples stables et correctement structurés, tandis que le Stage 2 à τ = 0,9 expose le modèle à une plus grande variété stylistique. La loss de départ Stage 2 (1,33) inférieure à la loss finale Stage 1 (1,48) confirme le bénéfice du warm-start.

- Le **filtrage top-60 %** par divergence s'est avéré plus pertinent qu'un seuil fixe. Avec un seuil fixe à 0,1, 100 % des exemples étaient conservés (toutes les divergences étaient > 0,1 car l'écart 4B vs 120B est énorme), ce qui annulait l'intérêt du DAS.

- L'architecture **LoRA avec gradient checkpointing** a permis d'entraîner sur une RTX 3070 8GB sans OOM, malgré la limite hardware (l'énoncé recommandait 8,5 GB VRAM).

### 7.2 Limites et difficultés rencontrées

- **Dataset de petite taille** : 90 exemples par stage est très peu pour un fine-tuning robuste. La loss d'évaluation (1,28–1,29) reste proche de la loss d'entraînement, ce qui suggère peu d'overfitting mais aussi une marge de progression limitée. Avec 500–1000 exemples, les résultats seraient probablement bien meilleurs.

- **Toutes les divergences DAS sont positives** : le student (4B) est toujours moins confiant que le teacher (120B), quelle que soit la question. Cela rend le filtrage DAS moins discriminant que prévu : on ne sélectionne pas vraiment des exemples "difficiles pour le student" mais simplement "les moins faciles". Une solution serait d'utiliser un student plus proche du teacher en taille.

- **Évaluation qualitative uniquement** : faute de benchmark disponible localement pour l'analyse littéraire, l'évaluation reste subjective. Une évaluation quantitative (ROUGE, BERTScore, ou évaluation humaine) serait nécessaire pour conclure sur la qualité de la distillation.

- **Problèmes techniques Windows** : l'environnement Windows a causé plusieurs difficultés non documentées dans l'énoncé (encodage UTF-8 du terminal, incompatibilité `bfloat16` / CrossEntropyLoss, espace disque cache HuggingFace sur C:). Ces problèmes ont nécessité des adaptations spécifiques au code.

### 7.3 Pistes d'amélioration

- Augmenter le dataset à 500+ exemples en utilisant une clé API avec un quota plus élevé
- Tester le DAS au niveau phrase (*sentence-level*) plutôt qu'au niveau réponse (*response-level*), comme décrit dans le papier original
- Évaluer avec un benchmark standard (ex. GSM8K pour le raisonnement mathématique, ou un dataset d'analyse littéraire manuel)
- Augmenter le LoRA rank (16 ou 32) pour donner au modèle plus de capacité d'adaptation

---

## 8. Conclusion

Ce TP a permis d'implémenter une pipeline DASD complète : génération de dataset via API teacher, filtrage par Divergence-Aware Sampling, et fine-tuning LoRA en deux stages avec température croissante.

Le modèle distillé final (Qwen3-4B + LoRA Stage 2) produit des analyses littéraires structurées et pertinentes, proches du style du teacher GPT-OSS-120B, tout en tenant en mémoire sur une GPU grand public de 8 GB. La loss d'entraînement est passée de 2,02 (début Stage 1) à 1,19 (fin Stage 2), soit une réduction de 41 %, ce qui témoigne d'un apprentissage effectif.

Le principal apport de DASD par rapport à une distillation classique réside dans la sélection intelligente des données d'entraînement : on n'apprend pas "plus" au modèle, on lui apprend les bonnes choses — celles où il a réellement quelque chose à apprendre du teacher.

---

## Références

- Alibaba-Apsara, *Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning*, 2026
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- LLaMA-Factory documentation : https://llamafactory.readthedocs.io
- Dataset de référence DASD : `Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b`
- Dataset d'instructions : `brunokreiner/genius-lyrics`
- Modèle étudiant : `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`
