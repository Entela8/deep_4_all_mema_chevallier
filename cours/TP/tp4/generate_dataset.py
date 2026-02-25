"""
TP4 - Phase 3 : Génération du dataset via API Infomaniak (Teacher)
Dataset source : brunokreiner/genius-lyrics
Tâche : Analyse littéraire de paroles de musique
"""

import json
import sys
import time
import random
from pathlib import Path
from datasets import load_dataset
import openai

# Fix encodage terminal Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = "nKuJabWS1epvq3x-m8by6NOU4xP4_znNL9OhmgXBPz9OeWOHlyGJIENnG8oXLT-4oOXNmESqExEMZv6o"  # <-- Remplacez par votre clé
BASE_URL = "https://api.infomaniak.com/2/ai/48/openai/v1"
TEACHER_MODEL = "openai/gpt-oss-120b"  # Modèle teacher sur Infomaniak

# Nombre d'exemples à générer par stage
N_STAGE1 = 150  # Basse température (stable)
N_STAGE2 = 150  # Haute température (diversité)

# Températures
TEMP_STAGE1 = 0.3
TEMP_STAGE2 = 0.9

# Filtres qualité sur les lyrics
MIN_LYRICS_LEN = 300    # Assez long pour analyser
MAX_LYRICS_LEN = 2000   # Pas trop long pour le prompt

# Dossier de sortie
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are an expert literary critic specializing in popular music.
When given song lyrics, you must reason through the analysis carefully step by step.

Always structure your response as follows:
1. First, reason inside <reasoning>...</reasoning> tags where you:
   - Identify the main themes and motifs
   - Analyze the poetic devices (metaphors, repetition, rhyme scheme, imagery)
   - Examine the emotional arc and narrative structure
   - Consider the cultural and artistic context
2. Then provide a concise, well-structured literary analysis as your final answer.

Be thorough in your reasoning but clear and insightful in your final analysis."""


# ============================================================
# CHARGEMENT ET FILTRAGE DU DATASET
# ============================================================

def load_lyrics_samples(n_total: int, seed: int = 42) -> list[dict]:
    """Charge et filtre des paroles de qualité depuis le dataset."""
    print("Chargement du dataset genius-lyrics...")
    ds = load_dataset("brunokreiner/genius-lyrics", split="train")

    # Filtrage : anglais, longueur correcte, avec artiste si possible
    filtered = []
    for ex in ds:
        lyrics = ex.get("lyrics", "") or ""
        if (
            ex.get("is_english", False)
            and MIN_LYRICS_LEN <= len(lyrics) <= MAX_LYRICS_LEN
            and len(lyrics.split()) >= 50  # Au moins 50 mots
        ):
            filtered.append({
                "lyrics": lyrics,
                "artist_name": ex.get("artist_name") or "Unknown Artist",
                "genres": ex.get("genres_list") or [],
            })

    print(f"Exemples éligibles : {len(filtered)}")

    # Échantillonnage reproductible
    random.seed(seed)
    sample = random.sample(filtered, min(n_total, len(filtered)))
    return sample


# ============================================================
# APPEL API TEACHER
# ============================================================

def build_user_prompt(artist: str, lyrics: str) -> str:
    """Construit le prompt utilisateur pour une chanson."""
    artist_info = f" by {artist}" if artist != "Unknown Artist" else ""
    return f"""Please provide a literary analysis of the following song lyrics{artist_info}:

---
{lyrics}
---

Analyze the themes, poetic devices, emotional arc, and narrative structure of these lyrics."""


def call_teacher_api(
    client: openai.OpenAI,
    user_prompt: str,
    temperature: float,
    max_retries: int = 3,
) -> dict | None:
    """
    Appelle l'API Infomaniak et retourne la réponse + logprobs.
    Retourne None en cas d'échec définitif.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=2000,
                logprobs=True,
                top_logprobs=1,
            )

            choice = response.choices[0]
            content = choice.message.content

            # Extraction des logprobs token par token
            logprobs_data = []
            if choice.logprobs and choice.logprobs.content:
                for token_logprob in choice.logprobs.content:
                    logprobs_data.append({
                        "token": token_logprob.token,
                        "logprob": token_logprob.logprob,
                    })

            return {
                "content": content,
                "logprobs": logprobs_data,
                "finish_reason": choice.finish_reason,
            }

        except openai.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"  Rate limit atteint. Attente {wait}s...")
            time.sleep(wait)
        except openai.APIError as e:
            print(f"  Erreur API (tentative {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
        except Exception as e:
            print(f"  Erreur inattendue : {e}")
            return None

    return None


# ============================================================
# GÉNÉRATION D'UN STAGE
# ============================================================

def generate_stage(
    client: openai.OpenAI,
    samples: list[dict],
    stage: int,
    temperature: float,
    output_file: Path,
) -> list[dict]:
    """Génère les réponses pour un stage et sauvegarde en JSON."""
    print(f"\n{'='*50}")
    print(f"STAGE {stage} - température={temperature}")
    print(f"Nombre d'exemples : {len(samples)}")
    print(f"{'='*50}")

    results = []
    llamafactory_data = []  # Format ShareGPT pour LLaMA-Factory

    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {sample['artist_name']} - {sample['genres'][:1]}")

        user_prompt = build_user_prompt(sample["artist_name"], sample["lyrics"])
        api_result = call_teacher_api(client, user_prompt, temperature)

        if api_result is None:
            print(f"  ÉCHEC - exemple ignoré")
            continue

        content = api_result["content"]

        # Vérification qualité minimale
        if len(content) < 200:
            print(f"  Réponse trop courte ({len(content)} chars) - ignorée")
            continue

        # Stockage complet (avec logprobs pour DAS)
        results.append({
            "stage": stage,
            "temperature": temperature,
            "artist_name": sample["artist_name"],
            "genres": sample["genres"],
            "lyrics": sample["lyrics"],
            "instruction": user_prompt,
            "response": content,
            "logprobs": api_result["logprobs"],
            "finish_reason": api_result["finish_reason"],
        })

        # Format ShareGPT pour LLaMA-Factory (sans logprobs)
        llamafactory_data.append({
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": user_prompt},
                {"from": "gpt", "value": content},
            ]
        })

        print(f"  OK - {len(content)} chars, {len(api_result['logprobs'])} tokens")

        # Pause pour respecter le rate limit
        time.sleep(1)

    # Sauvegarde
    raw_file = output_file.parent / f"stage{stage}_raw.json"
    llamafactory_file = output_file.parent / f"stage{stage}_llamafactory.json"

    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(llamafactory_file, "w", encoding="utf-8") as f:
        json.dump(llamafactory_data, f, ensure_ascii=False, indent=2)

    print(f"\nSauvegardé : {len(results)} exemples")
    print(f"  → {raw_file}")
    print(f"  → {llamafactory_file}")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    # Vérification de la clé API
    if API_KEY == "VOTRE_CLE_API_ICI":
        raise ValueError("Remplacez API_KEY par votre vraie clé Infomaniak !")

    # Client OpenAI compatible Infomaniak
    client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)

    # Chargement du dataset
    total_needed = N_STAGE1 + N_STAGE2
    all_samples = load_lyrics_samples(total_needed)

    # Séparation en deux pools (pas de chevauchement)
    samples_stage1 = all_samples[:N_STAGE1]
    samples_stage2 = all_samples[N_STAGE1:N_STAGE1 + N_STAGE2]

    # Génération Stage 1 (basse température - stable)
    stage1_results = generate_stage(
        client=client,
        samples=samples_stage1,
        stage=1,
        temperature=TEMP_STAGE1,
        output_file=OUTPUT_DIR / "stage1.json",
    )

    # Génération Stage 2 (haute température - diversité)
    stage2_results = generate_stage(
        client=client,
        samples=samples_stage2,
        stage=2,
        temperature=TEMP_STAGE2,
        output_file=OUTPUT_DIR / "stage2.json",
    )

    # Récapitulatif
    print(f"\n{'='*50}")
    print("RÉCAPITULATIF GÉNÉRATION")
    print(f"{'='*50}")
    print(f"Stage 1 (τ={TEMP_STAGE1}) : {len(stage1_results)} exemples")
    print(f"Stage 2 (τ={TEMP_STAGE2}) : {len(stage2_results)} exemples")
    print(f"Total : {len(stage1_results) + len(stage2_results)} exemples")
    print(f"\nFichiers sauvegardés dans : {OUTPUT_DIR}")
    print("\nProchaine étape : run das_filter.py pour appliquer le DAS")


if __name__ == "__main__":
    main()