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

    def decide_keep_prompt(self, teacher_answer, student_answer):
        teacher_logprob = teacher_answer.get("mean_logprob", 0.0)
        student_logprob = student_answer.get("mean_logprob", 0.0)

        print(teacher_logprob, student_logprob)

        divergence = teacher_logprob - student_logprob

        print(divergence)

    def run(self, prompt):
        print(f"Traitement du prompt : '{prompt}'")

        # 1. Teacher
        teacher_answer = self.get_teacher_data(prompt)
        if not teacher_answer:
            return

        print(f"Réponse Teacher reçue ({len(teacher_answer["content"])} chars).")

        # 2. Student & Calculs
        try:
            student_answer = self.get_student_logprobs(prompt, teacher_answer["content"])

            # 3. Décision
            return self.decide_keep_prompt(teacher_answer, student_answer)

        except Exception as e:
            print(f"Erreur durant le calcul DAS : {e}")
            import traceback
            traceback.print_exc()
            return None


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Clé API
    API_KEY = "token here"

    # ID Modèle Étudiant (Compatible 4-bit unsloth/bnb)
    STUDENT_ID = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"

    pipeline = DASPipelineQwen(openai_api_key=API_KEY, student_model_id=STUDENT_ID)

    # Test
    test_prompt = "Explique le principe de la supraconductivité de manière simple."
    result = pipeline.run(test_prompt)
