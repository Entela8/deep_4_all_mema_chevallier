"""
Application Gradio : Tournoi de la Guilde - Leaderboard

Interface web pour :
- Upload et evaluation automatique des modeles
- Leaderboard en temps reel avec SQLite
- Historique des soumissions

Usage:
    python app_leaderboard.py
    # Ouvre http://localhost:7860
"""

import hashlib
import json
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
import torch.nn as nn

from train import AdventurerDataset


def count_parameters(model: nn.Module) -> int:
    """Compte le nombre de parametres entrainables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_architecture_summary(model: nn.Module) -> dict:
    """Extrait un resume de l'architecture du modele."""
    summary = {
        'class_name': model.__class__.__name__,
        'n_params':   count_parameters(model),
        'layers':     []
        }
    for name, module in model.named_modules():
        if name:
            layer_info = {'name': name, 'type': module.__class__.__name__}
            if isinstance(module, nn.Linear):
                layer_info['shape'] = f"{module.in_features}->{module.out_features}"
            elif isinstance(module, nn.Dropout):
                layer_info['p'] = module.p
            summary['layers'].append(layer_info)
    return summary


# =============================================================================
# Configuration
# =============================================================================

DB_PATH = Path(__file__).parent / "leaderboard.db"
TEST_SECRET_PATH = Path(__file__).parent / "solution" / "test_secret.csv"
VAL_PATH = Path(__file__).parent / "data" / "val.csv"


# =============================================================================
# Base de donnees SQLite
# =============================================================================

def init_database():
    """Initialise la base de donnees SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions
            (
                id
                INTEGER
                PRIMARY
                KEY
                AUTOINCREMENT,
                team_name
                TEXT
                NOT
                NULL,
                model_hash
                TEXT
                NOT
                NULL,
                val_accuracy
                REAL,
                test_accuracy
                REAL,
                test_f1
                REAL,
                n_params
                INTEGER,
                architecture
                TEXT,
                submitted_at
                TIMESTAMP
                DEFAULT
                CURRENT_TIMESTAMP,
                is_best
                INTEGER
                DEFAULT
                0
            )
            """
            )

    cursor.execute(
            '''
            CREATE INDEX IF NOT EXISTS idx_team_name ON submissions(team_name)
            '''
            )

    conn.commit()
    conn.close()


def save_submission(
        team_name: str, model_hash: str, val_acc: float,
        test_acc: float, test_f1: float, n_params: int,
        architecture: dict
        ) -> int:
    """Sauvegarde une soumission dans la base."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Verifier si c'est le meilleur score de l'equipe
    cursor.execute(
            '''
            SELECT MAX(test_accuracy)
            FROM submissions
            WHERE team_name = ?
            ''', (team_name,)
            )
    best_score = cursor.fetchone()[0]

    is_best = 1 if best_score is None or test_acc > best_score else 0

    # Si c'est le meilleur, reset les autres
    if is_best:
        cursor.execute(
                '''
                UPDATE submissions
                SET is_best = 0
                WHERE team_name = ?
                ''', (team_name,)
                )

    cursor.execute(
            '''
            INSERT INTO submissions
            (team_name, model_hash, val_accuracy, test_accuracy, test_f1,
             n_params, architecture, is_best)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                team_name, model_hash, val_acc, test_acc, test_f1,
                n_params, json.dumps(architecture), is_best
                )
            )

    submission_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return submission_id


def get_leaderboard() -> pd.DataFrame:
    """Recupere le leaderboard (meilleur score par equipe)."""
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
            '''
            SELECT team_name                                      as "Equipe",
                   ROUND(val_accuracy * 100, 2)                   as "Val Acc (%)",
                   ROUND(test_accuracy * 100, 2)                  as "Test Secret (%)",
                   ROUND((val_accuracy - test_accuracy) * 100, 2) as "Gap (%)",
                   n_params                                       as "Params",
                   submitted_at                                   as "Soumis le"
            FROM submissions
            WHERE is_best = 1
            ORDER BY test_accuracy DESC
            ''', conn
            )

    conn.close()

    # Ajouter le rang
    if len(df) > 0:
        df.insert(0, 'Rang', range(1, len(df) + 1))

    return df


def get_stats() -> dict:
    """Recupere les statistiques globales."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(DISTINCT team_name) FROM submissions')
    n_teams = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM submissions')
    n_submissions = cursor.fetchone()[0]

    cursor.execute('SELECT MAX(test_accuracy) FROM submissions')
    best_score = cursor.fetchone()[0]

    cursor.execute('SELECT AVG(test_accuracy) FROM submissions WHERE is_best = 1')
    avg_score = cursor.fetchone()[0]

    conn.close()

    return {
        'n_teams':       n_teams or 0,
        'n_submissions': n_submissions or 0,
        'best_score':    best_score or 0,
        'avg_score':     avg_score or 0
        }


# =============================================================================
# Evaluation du modele
# =============================================================================

def load_model_from_file(model_path: str, input_dim: int = 12) -> tuple:
    """
    Charge un modele et retourne (model, architecture_info) ou (None, error).

    Supporte:
    - Modele complet (torch.save(model, path)) - NOUVEAU FORMAT
    - State dict avec architecture GuildOracle (retrocompatibilite)
    """
    try:
        # Charger le fichier (modele complet OU state_dict)
        loaded = torch.load(model_path, map_location='cpu', weights_only=False)

        if isinstance(loaded, nn.Module):
            # === MODELE COMPLET (nouveau format) ===
            model = loaded
            model.eval()

            # Tester le forward pass (batch=2 pour BatchNorm)
            x = torch.randn(2, input_dim)
            with torch.no_grad():
                output = model(x)

            # Verifier la sortie
            if output.dim() == 2 and output.shape[1] == 1:
                pass  # OK: (batch, 1)
            elif output.dim() == 1:
                pass  # OK: (batch,)
            else:
                return None, f"Sortie invalide: {output.shape}. Attendu: (batch, 1)"

            architecture = get_architecture_summary(model)
            return model, architecture

        else:
            return None, f"Type de fichier non reconnu: {type(loaded)}"

    except Exception as e:
        return None, f"Erreur de chargement: {str(e)}"


def evaluate_model(model: nn.Module, data_path: str) -> dict:
    """Evalue un modele sur un dataset."""
    data = AdventurerDataset(data_path, normalize=True)

    model.eval()
    with torch.no_grad():
        logits = model(data.features).squeeze()
        probs = torch.sigmoid(logits).numpy()
        predictions = (probs > 0.5).astype(int)
        labels = data.labels.numpy()

    accuracy = (predictions == labels).mean()

    # F1 score
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy':  float(accuracy),
        'f1':        float(f1),
        'precision': float(precision),
        'recall':    float(recall)
        }


def compute_model_hash(model_path: str) -> str:
    """Calcule le hash MD5 du modele."""
    with open(model_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


# =============================================================================
# Interface Gradio
# =============================================================================

def process_submission(team_name: str, model_file) -> tuple:
    """
    Traite une soumission : charge, evalue, sauvegarde.
    Retourne (message, leaderboard_df)
    """
    if not team_name or not team_name.strip():
        return "Erreur: Veuillez entrer un nom d'equipe.", get_leaderboard()

    team_name = team_name.strip()

    if model_file is None:
        return "Erreur: Veuillez uploader un fichier modele (.pt)", get_leaderboard()

    # Verifier que les datasets existent
    if not TEST_SECRET_PATH.exists():
        return "Erreur: Dataset test_secret.csv non trouve. Contactez l'enseignant.", get_leaderboard()

    if not VAL_PATH.exists():
        return "Erreur: Dataset val.csv non trouve.", get_leaderboard()

    try:
        # Gerer le fichier uploade
        model_path = model_file.name if hasattr(model_file, 'name') else model_file

        # Si c'est un ZIP, extraire
        if model_path.endswith('.zip'):
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)

                # Chercher le fichier .pt
                pt_files = list(Path(tmp_dir).rglob('*.pt'))
                if not pt_files:
                    return "Erreur: Aucun fichier .pt trouve dans le ZIP.", get_leaderboard()

                model_path = str(pt_files[0])
                return _evaluate_and_save(team_name, model_path)
        else:
            return _evaluate_and_save(team_name, model_path)

    except Exception as e:
        return f"Erreur: {str(e)}", get_leaderboard()


def _evaluate_and_save(team_name: str, model_path: str) -> tuple:
    """Evalue et sauvegarde le modele."""
    # Charger le modele
    model, config = load_model_from_file(model_path)

    if model is None:
        return f"Erreur: {config}", get_leaderboard()

    n_params = count_parameters(model)
    model_hash = compute_model_hash(model_path)

    # Evaluer sur validation
    val_results = evaluate_model(model, str(VAL_PATH))

    # Evaluer sur test secret
    test_results = evaluate_model(model, str(TEST_SECRET_PATH))

    # Sauvegarder
    submission_id = save_submission(
            team_name=team_name,
            model_hash=model_hash,
            val_acc=val_results['accuracy'],
            test_acc=test_results['accuracy'],
            test_f1=test_results['f1'],
            n_params=n_params,
            architecture=config
            )

    # Construire le message de resultat
    gap = val_results['accuracy'] - test_results['accuracy']

    # Préparer les valeurs formatées
    val_acc_str  = f"{val_results['accuracy']:.2%}"
    test_acc_str = f"{test_results['accuracy']:.2%}"
    gap_str      = f"{gap:+.2%}"
    f1_str       = f"{test_results['f1']:.2%}"

    # Déterminer le badge et la couleur (emojis et texte)
    if gap > 0.10:
        badge = "⚠️ **ATTENTION** : Gros écart ! Votre modèle sur-apprend."
        color = "🔴"
    elif gap > 0.05:
        badge = "⚙️ **Modéré** : Pensez à plus de régularisation."
        color = "🟡"
    else:
        badge = "✅ **Excellent** : Votre modèle généralise bien."
        color = "🟢"

    # Construction du message en Markdown
    message = f"""
**🏆 Résultats de votre soumission - ID: {submission_id}**

<div align="center">
<strong>{color} {badge}</strong>
</div>

### 📋 Détails de l'équipe
- **Nom:** `{team_name}`
- **Hash du modèle:** `{model_hash}`
- **Paramètres:** `{n_params:,}`

### 📊 Scores
| Métrique | Valeur |
| :--- | :--- |
| **Accuracy Validation** | `{val_acc_str}` |
| **Accuracy Test Secret**| `{test_acc_str}` |
| **Gap (Val - Test)** | `{gap_str}` |
| **F1-Score Test** | `{f1_str}` |
    """

    del model
    return message, get_leaderboard()


def refresh_leaderboard():
    """Rafraichit le leaderboard."""
    return get_leaderboard()


def get_stats_text():
    """Retourne les stats formatees."""
    stats = get_stats()
    return f"""
**Statistiques du Tournoi**
- Equipes participantes: {stats['n_teams']}
- Total soumissions: {stats['n_submissions']}
- Meilleur score: {stats['best_score']:.2%}
- Score moyen: {stats['avg_score']:.2%}
"""


# =============================================================================
# Application Gradio
# =============================================================================

def create_app():
    """Cree l'application Gradio."""

    # Initialiser la base
    init_database()

    with gr.Blocks(
            title="Tournoi de la Guilde - Leaderboard",
            theme=gr.themes.Soft()
            ) as app:
        gr.Markdown(
                """
                        # ⚔️ Tournoi de la Guilde des Aventuriers
                        ### Soumettez votre modele Oracle et grimpez dans le classement !
                        """
                )

        with gr.Tabs():
            # Tab 1: Soumission
            with gr.TabItem("📤 Soumettre"):
                with gr.Row():
                    with gr.Column(scale=1):
                        team_input = gr.Textbox(
                                label="Nom de l'equipe",
                                placeholder="Ex: Les Dragons de PyTorch",
                                max_lines=1
                                )
                        model_input = gr.File(
                                label="Modele (.pt ou .zip)",
                                file_types=[".pt", ".zip"],
                                type="filepath"
                                )
                        submit_btn = gr.Button(
                                "🚀 Soumettre",
                                variant="primary",
                                size="lg"
                                )

                    with gr.Column(scale=2):
                        result_output = gr.Markdown(
                                label="Resultat"
                                )

            # Tab 2: Leaderboard
            with gr.TabItem("🏆 Leaderboard"):
                gr.Markdown("### Classement en temps reel")

                stats_display = gr.Markdown(get_stats_text())

                leaderboard_table = gr.Dataframe(
                        value=get_leaderboard(),
                        label="Classement (meilleur score par equipe)",
                        interactive=False,
                        wrap=True
                        )

                refresh_btn = gr.Button("🔄 Rafraichir", size="sm")

            # Tab 3: Regles
            with gr.TabItem("📜 Regles"):
                gr.Markdown(
                        """
                                        ## Regles du Tournoi
                        
                                        ### Comment participer
                        
                                        1. **Entrainez** votre modele avec `train.py`
                                        2. **Soumettez** votre modele avec `submit.py`
                                        3. **Upload** Upload votre model le fichier .pt sur cette interface
                        
                                        ### Scoring
                        
                                        - Votre modele est evalue sur un **dataset secret**
                                        - Le classement est base sur l'**accuracy** du test secret
                                        - Seul votre **meilleur score** compte
                        
                                        ### Conseils
                        
                                        - Le dataset de test a une distribution **differente** !
                                        - Les modeles sur-appris seront penalises
                                        - Pensez a: Dropout, Weight Decay, Early Stopping
                        
                                        ### Anti-triche
                        
                                        - Chaque soumission est hashee
                                        - L'historique complet est conserve
                                        - Le dataset secret n'est jamais revele
                        
                                        ---
                                        *Que la chance soit avec vous, jeune Oracle !*
                                        """
                        )

        # Events
        submit_btn.click(
                fn=process_submission,
                inputs=[team_input, model_input],
                outputs=[result_output, leaderboard_table]
                )

        refresh_btn.click(
                fn=lambda: (get_leaderboard(), get_stats_text()),
                outputs=[leaderboard_table, stats_display]
                )

    return app


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Tournoi de la Guilde - Serveur Leaderboard")
    print("=" * 60)

    # Verifier les fichiers
    if not TEST_SECRET_PATH.exists():
        print(f"[!] ATTENTION: {TEST_SECRET_PATH} non trouve!")
        print("    Executez: python dataset.py")
    else:
        print(f"[OK] Dataset test secret: {TEST_SECRET_PATH}")

    if not VAL_PATH.exists():
        print(f"[!] ATTENTION: {VAL_PATH} non trouve!")
    else:
        print(f"[OK] Dataset validation: {VAL_PATH}")

    print(f"[OK] Base de donnees: {DB_PATH}")
    print("=" * 60)

    app = create_app()
    app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,  # Mettre True pour partager publiquement
            show_error=True
            )
