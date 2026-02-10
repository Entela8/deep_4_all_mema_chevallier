import numpy as np
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class DASPipelineQwen:
    def __init__(self, openai_api_key, student_model_id="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"):
        """
        Initialise le pipeline DAS avec un Teacher (API) et un Student (Local 4-bit).
        """
        # 1. Configuration Student (4-bit quantization)
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, )

        self.student_model_id = student_model_id
        print(f"Chargement du modèle étudiant : {self.student_model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.student_model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
                self.student_model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
                )
        self.model.eval()

        # 2. Configuration Teacher (OpenAI Compatible API - ex: Infomaniak)
        # Note: Remplacez base_url par l'URL correcte si différent de l'exemple
        self.client = OpenAI(
                api_key=openai_api_key, base_url="https://api.infomaniak.com/2/ai/48/openai/v1"
                )
        self.teacher_model_name = "openai/gpt-oss-120b"

    def get_teacher_data(self, user_prompt, temperature=0.7):
        """
        Génère la réponse du Teacher avec les logprobs.
        """
        messages = [{"role": "user", "content": user_prompt}]
        # Note: Assurez-vous que le modèle supporte logprobs=True
        response = self.client.chat.completions.create(
                model=self.teacher_model_name, messages=messages, temperature=temperature, logprobs=True, top_logprobs=1
                )

        content = response.choices[0].message.content
        logprobs_data = response.choices[0].logprobs
        tokens = []
        logprobs = []
        # On vérifie si logprobs est disponible (certaines API compatibles ne le renvoient pas)
        if logprobs_data:
            for token_info in logprobs_data.content:
                tokens.append(token_info.token)
                logprobs.append(token_info.logprob)
        else:
            raise ValueError("L'API Teacher n'a pas renvoyé de logprobs. Vérifiez la compatibilité.")

        # Compute total log probability (sum of logprobs)
        total_logprob = sum(logprobs) if logprobs else 0.0

        # Compute geometric mean of probabilities
        # P_geom = exp(mean(logprobs))
        mean_logprob = np.exp(np.mean(logprobs)) if logprobs else 0.0
        return {
            "content":      content, "tokens": tokens, "logprobs": logprobs, "total_logprob": total_logprob,
            "mean_logprob": mean_logprob, "num_tokens": len(tokens)
            }

    def get_student_logprobs(self, prompt: str, response: str) -> dict:
        """
        Calcule les log-probabilités de la réponse (Student) de manière robuste.
        Utilise la méthode de masquage standard (Labels = -100 pour le prompt).
        """
        # 1. Préparer le texte complet (Prompt + Réponse)
        # On utilise le chat template qui gère proprement les balises <|im_start|>, etc.
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
            ]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # 2. Tokenizer le tout
        # return_tensors='pt' nous donne directement les tenseurs PyTorch
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids

        # 3. Identifier la longueur du Prompt pour le masquage
        # On regénère le prompt SEUL avec l'amorce de réponse (add_generation_prompt=True)
        # Cela inclut "<|im_start|>assistant\n" à la fin, pour s'aligner parfaitement.
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
                )

        # On tokenise le prompt seul pour avoir sa longueur exacte en tokens
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        response_start_idx = prompt_tokens.shape[1]

        # 4. Créer les Labels (Masking du Prompt)
        # -100 est l'index ignoré par défaut par CrossEntropyLoss de PyTorch
        labels = input_ids.clone()
        # On masque tout ce qui est avant le début de la réponse
        labels[:, :response_start_idx] = -100

        # 5. Calcul "Clean" avec CrossEntropyLoss
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

            # Shift des logits et labels pour la prédiction "next token"
            # logits[t] prédit labels[t+1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # reduction='none' nous donne la perte pour chaque token individuel
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)

            # La Loss est par définition -log(p), donc log_prob = -loss
            token_logprobs = -token_losses

            # On ne garde que les tokens de la réponse (ceux qui n'étaient pas masqués à -100)
            # Note: shift_labels a été décalé, donc on utilise son masque
            valid_mask = shift_labels != -100
            valid_logprobs = token_logprobs[valid_mask].cpu().numpy()

        # Calcul des statistiques DAS
        total_logprob = np.sum(valid_logprobs)
        mean_logprob = np.exp(np.mean(valid_logprobs)) if len(valid_logprobs) > 0 else 0.0

        return {
            "total_logprob": total_logprob,
            "mean_logprob":  mean_logprob,
            "num_tokens":    len(valid_logprobs),
            "logprobs":      valid_logprobs.tolist()
            }

    # ----------------------------------------------------------------
    # PHASE 4 : Décision DAS
    # ----------------------------------------------------------------

    def decide_keep_prompt(self, teacher_answer, student_answer, threshold=0.1):
        """
        Décide si un exemple doit être conservé pour l'entraînement.

        Logique DAS (niveau réponse) :
          divergence = P_teacher - P_student   (probabilités moyennes géométriques)

          divergence > threshold  → TEACHER_SENTENCE → GARDER
          |divergence| <= threshold → SHARED_SENTENCE → NEUTRE (garder aussi)
          divergence < -threshold → STUDENT_SENTENCE → REJETER

        Retourne un dict avec le score et la décision.
        """
        teacher_prob = teacher_answer.get("mean_logprob", 0.0)
        student_prob = student_answer.get("mean_logprob", 0.0)
        divergence = teacher_prob - student_prob

        if divergence > threshold:
            label = "TEACHER_SENTENCE"
            keep = True
        elif divergence < -threshold:
            label = "STUDENT_SENTENCE"
            keep = False
        else:
            label = "SHARED_SENTENCE"
            keep = True

        return {
            "keep":         keep,
            "label":        label,
            "divergence":   divergence,
            "teacher_prob": teacher_prob,
            "student_prob": student_prob,
        }

    # ----------------------------------------------------------------
    # PHASE 4 : Filtrage d'un dataset complet
    # ----------------------------------------------------------------

    def filter_dataset(self, input_path, output_dir, stage, threshold=0.1):
        """
        Traite tous les exemples d'un fichier stage*_raw.json et filtre
        selon le score DAS.

        Produit :
          - stage{N}_filtered_raw.json      : exemples conservés (avec scores)
          - stage{N}_filtered_llamafactory.json : format prêt pour LLaMA-Factory
          - stage{N}_das_scores.png         : histogramme des divergences
        """
        import json
        import matplotlib.pyplot as plt
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Chargement des données générées (Phase 3)
        with open(input_path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        print(f"\n{'='*55}")
        print(f"FILTRAGE DAS - Stage {stage}")
        print(f"Exemples à traiter : {len(examples)}")
        print(f"Seuil de divergence : {threshold}")
        print(f"{'='*55}")

        filtered_raw = []
        llamafactory_data = []
        all_scores = []

        for i, ex in enumerate(examples):
            instruction = ex["instruction"]
            response    = ex["response"]
            artist      = ex.get("artist_name", "Unknown")
            print(f"[{i+1}/{len(examples)}] {artist[:30]:<30}", end=" ")

            try:
                # Calcul des logprobs student sur la réponse du teacher
                student_answer = self.get_student_logprobs(instruction, response)

                # Les logprobs du teacher sont déjà dans le fichier (Phase 3)
                teacher_logprobs = ex.get("logprobs", [])
                teacher_probs    = [lp["logprob"] for lp in teacher_logprobs]
                teacher_mean_prob = float(np.exp(np.mean(teacher_probs))) if teacher_probs else 0.0

                teacher_answer = {
                    "mean_logprob": teacher_mean_prob,
                    "content":      response,
                }

                # Décision DAS
                decision = self.decide_keep_prompt(teacher_answer, student_answer, threshold)
                all_scores.append(decision["divergence"])

                status = "KEEP  " if decision["keep"] else "SKIP  "
                print(f"{status} | div={decision['divergence']:+.4f} | {decision['label']}")

                if decision["keep"]:
                    # Données complètes avec scores (pour analyse)
                    filtered_raw.append({
                        **ex,
                        "das_divergence":   decision["divergence"],
                        "das_label":        decision["label"],
                        "das_teacher_prob": decision["teacher_prob"],
                        "das_student_prob": decision["student_prob"],
                    })

                    # Format ShareGPT pour LLaMA-Factory
                    llamafactory_data.append({
                        "conversations": [
                            {"from": "human", "value": instruction},
                            {"from": "gpt",   "value": response},
                        ]
                    })

            except Exception as e:
                print(f"ERREUR : {e}")
                continue

        # ── Sauvegarde des résultats ──────────────────────────────────
        raw_out   = output_dir / f"stage{stage}_filtered_raw.json"
        lmf_out   = output_dir / f"stage{stage}_filtered_llamafactory.json"
        plot_out  = output_dir / f"stage{stage}_das_scores.png"

        with open(raw_out, "w", encoding="utf-8") as f:
            json.dump(filtered_raw, f, ensure_ascii=False, indent=2)

        with open(lmf_out, "w", encoding="utf-8") as f:
            json.dump(llamafactory_data, f, ensure_ascii=False, indent=2)

        # ── Histogramme des scores DAS ────────────────────────────────
        if all_scores:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(all_scores, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
            ax.axvline(x= threshold, color="green",  linestyle="--", linewidth=1.5,
                       label=f"Seuil KEEP (+{threshold})")
            ax.axvline(x=-threshold, color="red",    linestyle="--", linewidth=1.5,
                       label=f"Seuil REJECT (-{threshold})")
            ax.axvline(x=0,          color="orange", linestyle=":",  linewidth=1.2,
                       label="Divergence = 0")
            ax.set_xlabel("Divergence (P_teacher − P_student)", fontsize=12)
            ax.set_ylabel("Nombre d'exemples", fontsize=12)
            ax.set_title(f"Distribution DAS — Stage {stage}", fontsize=14)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_out, dpi=150)
            plt.close(fig)

        # ── Récapitulatif ─────────────────────────────────────────────
        n_kept     = len(filtered_raw)
        n_total    = len(examples)
        keep_rate  = 100 * n_kept / n_total if n_total else 0
        mean_div   = float(np.mean(all_scores)) if all_scores else 0.0

        print(f"\n── Résultats Stage {stage} ──────────────────────────────")
        print(f"  Conservés  : {n_kept}/{n_total} ({keep_rate:.1f}%)")
        print(f"  Rejetés    : {n_total - n_kept}/{n_total}")
        print(f"  Divergence moyenne : {mean_div:+.4f}")
        print(f"  → {raw_out}")
        print(f"  → {lmf_out}")
        print(f"  → {plot_out}")

        return filtered_raw

    # ----------------------------------------------------------------
    # run() : traitement d'un seul prompt (inchangé + décision complète)
    # ----------------------------------------------------------------

    def run(self, prompt):
        print(f"Traitement du prompt : '{prompt[:60]}...'")

        teacher_answer = self.get_teacher_data(prompt)
        if not teacher_answer:
            return None

        print(f"Réponse Teacher reçue ({len(teacher_answer['content'])} chars).")

        try:
            student_answer = self.get_student_logprobs(prompt, teacher_answer["content"])
            decision = self.decide_keep_prompt(teacher_answer, student_answer)
            print(f"Décision : {decision['label']} | divergence={decision['divergence']:+.4f} | keep={decision['keep']}")
            return decision
        except Exception as e:
            print(f"Erreur durant le calcul DAS : {e}")
            import traceback
            traceback.print_exc()
            return None


# ====================================================================
# MAIN EXECUTION — Phase 4 : filtrage du dataset généré
# ====================================================================
if __name__ == "__main__":
    from pathlib import Path

    API_KEY    = "nKuJabWS1epvq3x-m8by6NOU4xP4_znNL9OhmgXBPz9OeWOHlyGJIENnG8oXLT-4oOXNmESqExEMZv6o"
    STUDENT_ID = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"

    # Dossier contenant les fichiers stage*_raw.json générés en Phase 3
    DATA_DIR   = Path(__file__).parent.parent / "tp4" / "data"
    OUTPUT_DIR = Path(__file__).parent / "data_filtered"

    DAS_THRESHOLD = 0.1  # Seuil de divergence pour KEEP / REJECT

    # Chargement unique du pipeline (student model + API)
    pipeline = DASPipelineQwen(openai_api_key=API_KEY, student_model_id=STUDENT_ID)

    # Filtrage Stage 1 et Stage 2
    for stage in [1, 2]:
        raw_file = DATA_DIR / f"stage{stage}_raw.json"
        if not raw_file.exists():
            print(f"[ATTENTION] Fichier introuvable : {raw_file}")
            print("  → Lancez d'abord generate_dataset.py (Phase 3)")
            continue

        pipeline.filter_dataset(
            input_path=raw_file,
            output_dir=OUTPUT_DIR,
            stage=stage,
            threshold=DAS_THRESHOLD,
        )

    print("\nPhase 4 terminée.")
    print(f"Fichiers prêts pour LLaMA-Factory dans : {OUTPUT_DIR}")
