import torch
import numpy as np
import nltk
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Nécessaire pour séparer les phrases
nltk.download('punkt', quiet=True)


class DASPipelineQwen:
    def __init__(self, openai_api_key):
        # Configuration pour charger le modèle en 4-bit via transformers
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                )

        model_id = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
        # Note: "Qwen3" n'est pas encore sorti officiellement au moment de mes données,
        # j'utilise ici un ID Qwen 2.5 4-bit très performant comme placeholder pour votre ID spécifique.
        # Remplacez par votre ID exact : "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"

        self.model_id = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # Mettez votre ID ici

        print(f"Chargement du modèle étudiant : {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
                )
        self.model.eval()

        # Client OpenAI (Teacher)
        self.client = OpenAI(api_key=openai_api_key)

    def get_teacher_data(self, user_prompt, temperature=1.0):
        """
        1. Appelle le Teacher (OpenAI) pour générer la réponse et les logprobs.
        """
        messages = [{"role": "user", "content": user_prompt}]

        response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=temperature,
                logprobs=True,  # Important
                top_logprobs=1
                )

        content = response.choices[0].message.content
        logprobs = response.choices[0].logprobs.content
        return content, logprobs

    def calculate_sentence_scores(self, user_prompt, teacher_text, teacher_logprobs):
        """
        Calcule P_Teacher et P_Student pour chaque phrase.
        Cette fonction gère l'alignement complexe entre les tokens OpenAI,
        les tokens Qwen, et les phrases.
        """

        # --- A. PRÉPARATION DU PROMPT ÉTUDIANT ---
        messages = [{"role": "user", "content": user_prompt}]

        # On formatte le prompt comme le modèle s'attend à le voir
        prompt_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # Ajoute le header pour le début de la réponse assistant
                )

        # Le texte complet que l'étudiant "voit" pour s'entrainer est : Prompt Formatté + Réponse Teacher
        full_input_str = prompt_str + teacher_text

        # --- B. CALCUL DES PROBS ÉTUDIANT (Forward Pass unique) ---
        inputs = self.tokenizer(full_input_str, return_tensors="pt").to(self.model.device)

        # On veut savoir où commence la réponse dans les tokens de l'étudiant
        # pour ne calculer la loss que sur la réponse (pas sur le prompt utilisateur)
        prompt_tokens_len = len(self.tokenizer(prompt_str, add_special_tokens=False)["input_ids"])

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: (1, seq_len, vocab_size)

        # Shift logits pour obtenir la prob du token suivant
        # Logits[i] prédit Input[i+1]
        shift_logits = logits[0, :-1, :]
        shift_labels = inputs["input_ids"][0, 1:]

        # Calcul des log-probabilités pour chaque token de la séquence
        # CrossEntropyLoss(reduction='none') nous donne la loss par token
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(shift_logits, shift_labels)
        # log_prob = -loss
        token_logprobs_student = -token_losses.cpu().numpy()

        # On ne garde que la partie "Réponse" (on ignore le prompt)
        # Attention aux index : shift_labels[i] correspond au token inputs[i+1]
        # Si prompt a une longueur N, le premier token de réponse est à l'index N.
        # Dans shift_labels, c'est l'index N-1.
        response_logprobs_student = token_logprobs_student[prompt_tokens_len - 1:]

        # Récupérons aussi les IDs des tokens de la réponse pour reconstruction/alignement
        response_token_ids = shift_labels[prompt_tokens_len - 1:].cpu().numpy()

        # --- C. DÉCOUPAGE EN PHRASES ET ALIGNEMENT ---
        # C'est la partie délicate : aligner les scores OpenAI (tokens A)
        # et les scores Qwen (tokens B) sur les mêmes phrases (texte).

        sentences = nltk.tokenize.sent_tokenize(teacher_text)

        results = []

        # Curseurs pour suivre notre position dans les listes de logprobs
        openai_cursor = 0
        qwen_cursor = 0

        for sent in sentences:
            # 1. Score Teacher (OpenAI) pour cette phrase
            # On avance dans les logprobs OpenAI jusqu'à couvrir la phrase
            current_sent_accum = ""
            sent_openai_logprobs = []

            while openai_cursor < len(teacher_logprobs):
                t_data = teacher_logprobs[openai_cursor]
                sent_openai_logprobs.append(t_data.logprob)
                current_sent_accum += t_data.token
                openai_cursor += 1
                # Check simple de fin de phrase (peut être amélioré avec des offsets précis)
                if len(current_sent_accum) >= len(sent):
                    break

            p_teacher = np.exp(np.mean(sent_openai_logprobs)) if sent_openai_logprobs else 0

            # 2. Score Student (Qwen) pour cette phrase
            # On décode les tokens Qwen un par un pour voir quand on finit la phrase
            current_sent_accum_qwen = ""
            sent_qwen_logprobs = []

            while qwen_cursor < len(response_token_ids):
                tid = response_token_ids[qwen_cursor]
                # Décodage partiel
                token_str = self.tokenizer.decode([tid])
                sent_qwen_logprobs.append(response_logprobs_student[qwen_cursor])
                current_sent_accum_qwen += token_str
                qwen_cursor += 1

                # Check si on a couvert la phrase
                # Note: Le tokenizer Qwen peut ajouter des espaces, il faut nettoyer un peu pour comparer
                if len(current_sent_accum_qwen) >= len(sent):
                    break

            p_student = np.exp(np.mean(sent_qwen_logprobs)) if sent_qwen_logprobs else 0

            results.append(
                    {
                        "sentence":   sent,
                        "p_teacher":  p_teacher,
                        "p_student":  p_student,
                        "divergence": p_teacher - p_student
                        }
                    )

        return results

    def run_das(self, prompt):
        print(f"--- DAS Processing for: '{prompt}' ---")

        # 1. Génération
        text, teacher_logs = self.get_teacher_data(prompt)

        # 2. Calcul des scores
        try:
            stats = self.calculate_sentence_scores(prompt, text, teacher_logs)

            # 3. Affichage / Filtrage
            for i, s in enumerate(stats):
                divergence = s['p_teacher'] - s['p_student']
                # Critère DAS : Le teacher est confiant, l'étudiant non.
                is_teacher_forced = (s['p_teacher'] > 0.6) and (divergence > 0.2)
                tag = "✅ TEACHER_SENTENCE" if is_teacher_forced else ".."

                print(f"[{i}] T={s['p_teacher']:.2f} S={s['p_student']:.2f} | Diff={divergence:.2f} {tag}")
                print(f"    Let: {s['sentence'][:50]}...")

        except Exception as e:
            print(f"Erreur d'alignement (fréquent avec les tokens spéciaux) : {e}")
            # En production, il faut un alignement par char-offset plus robuste (ex: bibliothèque `token alignments`)


# --- MAIN ---
if __name__ == "__main__":
    # Remplacer par votre clé
    api_key = "sk-..."

    das = DASPipelineQwen(api_key)
    das.run_das("Explique pourquoi le ciel est bleu de manière scientifique.")