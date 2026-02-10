# TP4 : Distillation de Modèles de Raisonnement (DASD)

## Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning

---

## Hardware

- [Google Colab avec 1 T4 GPU gratuit](https://colab.research.google.com/)
- [Kaggle avec 2xT4 GPU gratuit](https://www.kaggle.com/)
- GPU Nvidia avec 8.5 GB de VRAM minimum

> **Note :** Pour Kaggle, validez votre email + numéro de téléphone pour accéder aux GPU.

---

## Ressources

- [LLamafactory Documentation](https://llamafactory.readthedocs.io/en/latest/)
- [Repo officiel du papier scientifique](https://github.com/D2I-ai/dasd-thinking)
- [Dataset de référence DASD](https://huggingface.co/datasets/Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b)
- [Multi GPU train pour kaggle](https://llamafactory.readthedocs.io/en/latest/advanced/distributed.html#id4)
- [Format de data pour LLamafactory](https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html#sharegpt)

### API Enseignant (Teacher)

Utilisez l'API AI d'Infomaniak :

- **URL** : `https://api.infomaniak.com/2/ai/48/openai/v1`
- **Modèles disponibles** : Consultez la documentation Infomaniak pour choisir votre modèle
- **Format** : Compatible OpenAI

---

## Objectifs

1. **Comprendre** la distillation de connaissances appliquée aux LLMs
2. **Générer** un dataset de raisonnement via API (temperatures variées)
3. **Implémenter** le Divergence-Aware Sampling (DAS)
4. **Entraîner** un modèle étudiant avec Llama-Factory
5. **Évaluer** les performances du modèle distillé

---

## Contexte Scientifique

### Le Problème

Les grands LLMs (GPT-4, Claude, Qwen-235B) ont d'excellentes capacités de raisonnement mais sont coûteux et impossibles
à déployer localement.

**Question :** Comment transférer les capacités de raisonnement d'un modèle "enseignant" massif vers un modèle "
étudiant" compact ?

### La Solution DASD

Le papier propose deux techniques clés :

| Technique                       | Objectif                                                     |
|---------------------------------|--------------------------------------------------------------|
| Temperature-Scheduled Learning  | Équilibrer stabilité (τ basse) et diversité (τ haute)        |
| Divergence-Aware Sampling (DAS) | Sélectionner les données où l'étudiant a le plus à apprendre |

---

## Stack Technique

| Composant       | Choix                                             |
|-----------------|---------------------------------------------------|
| Framework       | Llama-Factory                                     |
| Modèle Étudiant | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Fine-tuning     | LoRA                                              |
| API Teacher     | Infomaniak AI                                     |

---

# JOURNÉE 1 : Setup, Génération Dataset & DAS

## Phase 1 : Installation de l'environnement

Installez Llama-Factory sur Colab/Kaggle/local.

Ressources :

- [Installation Llama-Factory](https://llamafactory.readthedocs.io/en/latest/getting_started/installation.html)
- [Notebook de démo](https://colab.research.google.com/drive/1qy4thB5CLOVSxAY7VYlL4TsR3I9UOKuR?usp=sharing)

**Checkpoint :** `llamafactory-cli version` fonctionne sans erreur.

---

## Phase 2 : Étude du Dataset de référence

Explorez le dataset officiel DASD sur HuggingFace pour comprendre :

- Le format des données (instruction/response)
- La structure `<think>...</think>` ou `<reasoning>...</reasoning>` (si vous utiliser un teacher thinking)
- La longueur et qualité des réponses

```python
from datasets import load_dataset

reference_dataset = load_dataset(
        "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
        "stage1"
        )
```

---

## Phase 3 : Génération de votre Dataset via API

### 3.1 Choisir un dataset d'instructions source

Sélectionnez des instructions/questions (sans réponses) depuis un dataset existant :

- GSM8K (math)
- Alpaca (général)
- CodeAlpaca (code)
- **Ou créez les vôtres**

### 3.2 Configurer l'API Infomaniak

```python
import openai

client = openai.OpenAI(
        base_url="https://api.infomaniak.com/2/ai/48/openai/v1",
        api_key="VOTRE_CLE_API"
        )

# Choisissez un modèle disponible via l'API
TEACHER_MODEL = "..."  # À vous de déterminer
```

### 3.3 Générer les réponses

Implémentez une fonction qui :

1. Appelle l'API avec un **system prompt** encourageant le raisonnement structuré
1. exporter les logprobs pour appliquer DASD
1. Génère des réponses à **basse température** (τ ≈ 0.3) → Stage 1
1. Génère des réponses à **haute température** (τ ≈ 0.9) → Stage 2
1. Sauvegarde les résultats au format JSON

**À implémenter :**

- Gestion des erreurs et retry
- Filtrage de qualité des réponses
- Conversion au format Llama-Factory (ShareGPT ou Alpaca)

````json
{
  "model": "qwen3",
  "max_tokens": 5000,
  "top_logprobs": 1,
  "logprobs": true,
  "temperature": 0.15,
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant that reasoning step by step. Always structure your reasoning inside <reasoning>...</reasoning> tags before giving your final answer. Be thorough in your reasoning process."
    },
    {
      "role": "user",
      "content": "Traduire la phrase en EN: LE chat vas très bien !"
    }
  ]
}
````

---

## Phase 4 : Implémentation du Divergence-Aware Sampling (DAS)

Une implémentation de l'algo expliquer dans le papier scientifique [implementation_dasd.py](implementation_dasd.py)

### Concept Révisé

Le DAS ne juge pas une réponse dans sa globalité, mais analyse la **divergence phrase par phrase**. L'objectif est d'identifier et de conserver les réponses riches en "Teacher Sentences" : des étapes de raisonnement où le professeur est confiant mais où l'étudiant échoue ou hésite.

### Formule (Niveau Phrase)

Pour chaque phrase de la réponse, on calcule la divergence sur les probabilités linéaires (et non les log-probs bruts) :

*Où est la moyenne géométrique des probabilités des tokens de la phrase :*


### Matrice de Décision

L'algorithme classe chaque phrase pour décider de la qualité globale de l'exemple d'entraînement :

| Type de Phrase | Condition mathématique          | Signification | Action |
|---|---------------------------------|---|---|
| **Teacher Sentence** | *Pt >> PS* (ex: $0.9$ vs $0.3$) | Le Teacher sait, l’Étudiant ignore. C’est ici que réside la valeur pédagogique. | **GARDER** (signal fort) |
| **Shared Sentence** | *PT ~= PS* (ex: $0.9$ vs $0.8$) | Connaissances déjà acquises ou triviales. | **NEUTRE** (sert de liaison) |
| **Student Sentence** | *Ps > PT* (ex: $0.4$ vs $0.9$)                      | L’étudiant est « trop » confiant (hallucination probable) ou le Teacher est incertain. | **REJETER** (bruit nuisible) |

**Critère de filtrage final :** On conserve la réponse complète uniquement si elle contient une densité suffisante de *Teacher Sentences* (divergence positive significative).

### À implémenter

1. Charger le modèle étudiant (Qwen3-4B)
2. Calculer les log-probabilités des réponses générées
3. Filtrer/trier les exemples selon leur score DAS
4. Analyser la distribution des scores (histogramme)

---

## Phase 5 : Configuration de l'entraînement

Crée vos config pour le Stage 1 (données basse température) & Stage 2 (données haute température)

Faire attention de bien charger l'adapter lora entre le stage 1 & 2

Référence : [Configuration Llama-Factory](https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html)

---

# JOURNÉE 2 : Analyse et Évaluation

## Phase 7 : Vérification des résultats

- Vérifier que les checkpoints existent
- Tracer les courbes de loss
- Comparer Stage 1 vs Stage 2

---

## Phase 8 : Test du modèle distillé

1. Charger le modèle de base + adaptateur LoRA
2. Tester sur des prompts
3. Vérifier les réponses

---

## Phase 9 : Évaluation quantitative

Évaluez votre modèle sur un benchmark (ex: GSM8K) :

- Comparer avec le modèle de base (sans distillation)
- Mesurer l'amélioration

---

## Phase 10 : Documentation

Préparez votre rapport avec :

- Méthodologie et choix techniques
- Statistiques du dataset généré
- Courbes de loss
- Résultats d'évaluation
- Discussion et limites

---

## Livrables

### Rapport (PDF, 4-6 pages)

1. **Introduction** : Problématique et approche DASD
2. **Méthodologie** : Génération dataset, DAS, configuration entraînement
3. **Résultats** : Statistiques, courbes, exemples de sorties
4. **Discussion** : Comparaison avant/après, limites, améliorations

### Code & Données

- Notebook commenté et exécutable
- Dataset généré (stage1 et stage2)
- Fichiers de configuration YAML
- Implémentation DAS

---

## Critères d'Évaluation

| Critère                       | Points |
|-------------------------------|--------|
| Installation et configuration | 2      |
| Génération dataset via API    | 5      |
| Implémentation DAS            | 3      |
| Entraînement fonctionnel      | 4      |
| Évaluation et analyse         | 3      |
| Qualité du rapport            | 3      |
| **Total**                     | **20** |

---

## Troubleshooting

| Problème                               | Solution                                               |
|----------------------------------------|--------------------------------------------------------|
| Rate limit API                         | Augmenter le délai entre requêtes                      |
| OOM GPU                                | Réduire `cutoff_len`, activer `gradient_checkpointing` |
| Loss ne descend pas                    | Vérifier le format des données                         |
| Pas de `<reasoning>` dans les réponses | Ajuster le system prompt                               |

---

*Basé sur le papier "Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning" (Alibaba, 2026)*
