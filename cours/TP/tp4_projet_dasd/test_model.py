import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
ADAPTER_DIR = "saves/stage2_lora"   # adaptateur final (Stage 2)

TEST_LYRICS = [
    # Exemple 1 — pop classique
    """roses are red violets are blue
i never thought i'd fall so hard for you
every morning feels like brand new light
holding your hand makes everything feel right""",

    # Exemple 2 — rock contemplatif
    """standing at the edge of everything i know
the wind is cold but still i cannot go
i built these walls to keep the darkness out
but now they keep me in without a doubt""",
]

# ── Chargement du modèle ──────────────────────────────────────────────────────
print("Chargement du modele de base (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Chargement de l'adaptateur LoRA : {ADAPTER_DIR}...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()
print("Modele pret !\n")


# ── Inference ─────────────────────────────────────────────────────────────────
def analyze(lyrics: str, max_new_tokens: int = 512) -> str:
    prompt = (
        "Please provide a literary analysis of the following song lyrics:\n\n"
        f"---\n{lyrics}\n---\n\n"
        "Analyze the themes, poetic devices, emotional arc, and narrative structure of these lyrics."
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Desactiver le thinking mode Qwen3
    text += "<think>\n\n</think>\n\n"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy pour la reproductibilite
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Garder uniquement les tokens generes (pas le prompt)
    generated = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── Tests ─────────────────────────────────────────────────────────────────────
for i, lyrics in enumerate(TEST_LYRICS, 1):
    print("=" * 65)
    print(f"TEST {i}")
    print("=" * 65)
    print("Paroles :")
    print(lyrics)
    print("\nAnalyse du modele distille :")
    print("-" * 40)
    result = analyze(lyrics)
    print(result)
    print()
