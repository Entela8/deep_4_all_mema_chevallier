"""
Modèle Baseline : Oracle de la Guilde (Non Optimal)

Ce modèle est VOLONTAIREMENT non optimal.
Les étudiants doivent identifier et corriger les problèmes !

Contient:
- GuildOracle : MLP pour prédiction de survie (stats → survie)
- DungeonOracle : LSTM pour prédiction de survie (séquence d'événements → survie)
"""

import torch
import torch.nn as nn


# ============================================================================
# TP2 : Modèle MLP pour stats d'aventuriers
# ============================================================================


class GuildOracle(nn.Module):
    """
    Modèle baseline pour prédire la survie des aventuriers.

    Architecture : MLP profond (trop profond !)
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 256, num_layers: int = 5):
        """
        Args:
            input_dim: Nombre de features (8 stats)
            hidden_dim: Dimension des couches cachées
            num_layers: Nombre de couches cachées
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Couche de sortie
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor de shape (batch_size, input_dim)

        Returns:
            Logits de shape (batch_size, 1)
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les probabilités de survie."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les prédictions binaires."""
        proba = self.predict_proba(x)
        return (proba > 0.5).float()


# ============================================================================
# TP3 : Modèle LSTM pour séquences de donjon
# ============================================================================


class DungeonOracle(nn.Module):
    """
    Modèle baseline pour prédire la survie à partir d'une séquence d'événements.

    Architecture : Embedding + LSTM + Classifier

    PROBLEMES VOLONTAIRES (à corriger par les étudiants):
    1. Embedding dimension trop petite (8) -> perd de l'information semantique
    2. Un seul layer LSTM -> difficile de capturer les patterns complexes
    3. Pas de Dropout -> risque d'overfitting
    4. Utilise RNN simple au lieu de LSTM -> vanishing gradient sur longues sequences
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 2,
            hidden_dim: int = 258,
            num_layers: int = 1,
            dropout: float = 0.0,
            mode: str = "linear",
            bidirectional: bool = False,
            padding_idx: int = 0,
            max_length: int = 140
            ):
        """
        Args:
            vocab_size: Taille du vocabulaire (nombre d'événements uniques)
            embed_dim: Dimension des embeddings
            hidden_dim: Dimension de l'état caché du RNN/LSTM
            num_layers: Nombre de couches RNN/LSTM
            dropout: Dropout entre les couches (si num_layers > 1)
            mode: lstm, rnn, default: linear
            bidirectional: Si True, RNN bidirectionnel
            padding_idx: Index du token de padding (ignoré dans les embeddings)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.mode = mode.lower().strip()

        # Couche d'embedding : transforme les IDs en vecteurs denses
        # Le padding_idx=0 fait que le vecteur pour <PAD> reste à zéro
        self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=padding_idx
                )

        # Approche Baseline Linéaire (Alternative au RNN)
        # On aplatit tout : (Batch, Seq_Len * Embed_Dim)
        self.solo_embeddings = nn.Sequential(
                nn.Flatten(),
                nn.Linear(max_length * embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)  # Sortie directe pour comparaison
                )
        if self.mode != "linear":
            # Couche récurrente
            # PROBLEME: Par défaut c'est un RNN simple qui souffre du vanishing gradient
            rnn_class = nn.LSTM if self.mode == "lstm" else nn.RNN

            self.rnn = rnn_class(
                    input_size=embed_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional
                    )

            # Couche de classification
            # Si bidirectionnel, on a 2x hidden_dim
            classifier_input_dim = hidden_dim * 2 if bidirectional else hidden_dim

            self.classifier = nn.Sequential(
                    nn.Linear(classifier_input_dim, 1)
                    )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor de shape (batch_size, seq_length) contenant les IDs d'événements
            lengths: Tensor de shape (batch_size,) contenant les longueurs réelles
                     (optionnel, pour ignorer le padding)

        Returns:
            Logits de shape (batch_size, 1)
        """
        batch_size = x.size(0)

        # Étape 1: Embedding
        # (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        if self.mode != "linear":
            # Étape 2: Passage dans le RNN/LSTM
            # output: (batch, seq_len, hidden_dim * num_directions)
            # hidden: (num_layers * num_directions, batch, hidden_dim)
            if self.mode == "lstm":
                output, (hidden, cell) = self.rnn(embedded)
            else:
                output, hidden = self.rnn(embedded)

            # Étape 3: Extraire le dernier état caché
            # Pour un RNN standard, on prend la dernière sortie
            if self.bidirectional:
                # Concaténer forward et backward
                hidden_forward = hidden[-2]  # Dernière couche, direction forward
                hidden_backward = hidden[-1]  # Dernière couche, direction backward
                final_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
            else:
                # Juste la dernière couche
                final_hidden = hidden[-1]

            # Étape 4: Classification
            logits = self.classifier(final_hidden)

            return logits
        else:
            return self.solo_embeddings(embedded)

    def predict_proba(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """Retourne les probabilités de survie."""
        with torch.no_grad():
            logits = self.forward(x, lengths)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """Retourne les prédictions binaires."""
        proba = self.predict_proba(x, lengths)
        return (proba > 0.5).float()

    def get_embeddings(self) -> torch.Tensor:
        """Retourne les poids de la couche d'embedding pour visualisation."""
        return self.embedding.weight.detach().clone()


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Compte le nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module):
    """Affiche un résumé du modèle."""
    print("=" * 50)
    print("Résumé du modèle")
    print("=" * 50)
    print(model)
    print("-" * 50)
    print(f"Nombre de paramètres : {count_parameters(model):,}")
    print("=" * 50)
