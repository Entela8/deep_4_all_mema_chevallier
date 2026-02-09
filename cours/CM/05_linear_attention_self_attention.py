import torch
import torch.nn.functional as F

# Configuration
N = 1000  # Longueur de la séquence (ex: 1000 mots)
D = 64  # Dimension du modèle (ex: chaque mot est un vecteur de 64 nombres)

# Données simulées
Q = torch.randn(N, D)
K = torch.randn(N, D)
V = torch.randn(N, D)

print(f"--- Configuration: Séquence N={N}, Dimension D={D} ---\n")


# --- 1. TRANSFORMER STANDARD (Lourd) ---
def standard_attention_demo(q, k, v):
    print("1. [Standard] Calcul de Q x K.T")

    # Transposer D x N
    k_t = k.transpose(-1, -2)
    # On crée une matrice de scores N x N
    scores = torch.matmul(q, k_t)
    print(f"   -> Matrice d'Attention créée : {scores.shape} (soit {N * N} valeurs !)")

    # Softmax standard
    attn_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, v)
    return output


# --- 2. LINEAR TRANSFORMER (Léger) ---
def linear_attention_demo(q, k, v):
    print("\n2. [Linear] Projection phi(Q) et phi(K)")

    # --- LA CLÉ EST ICI : LE FEATURE MAP (Phi) ---
    # On utilise ELU + 1 pour s'assurer que toutes les valeurs sont POSITIVES (>0)
    # C'est ce qui remplace le rôle "positivant" du Softmax.
    # https://arxiv.org/abs/1511.07289
    """ELU(x)={
                x, if x>0
                α∗(exp(x)−1), if x≤0
            }
 
        """
    q_prime = F.elu(q) + 1
    k_prime = F.elu(k) + 1

    print(f"   -> q_prime shape : {q_prime.shape}")
    print(f"   -> k_prime shape : {k_prime.shape}")

    # Changement d'ordre de multiplication : (K.T x V)
    # On calcule d'abord le "contexte global" ou "mémoire"
    print("   -> Calcul de K.T x V (Compression)")
    kv_matrix = torch.matmul(k_prime.transpose(-1, -2), v)

    # NOTEZ LA DIFFÉRENCE DE TAILLE
    print(f"   -> Matrice Intermédiaire créée : {kv_matrix.shape} (Seulement {D * D} valeurs !)")

    # Normalisation (car on n'a pas fait de Softmax)
    z = torch.matmul(q_prime, k_prime.sum(dim=0).unsqueeze(1))

    # Application de la Query sur le contexte pré-calculé
    output = torch.matmul(q_prime, kv_matrix) / z
    return output


# Exécution
_ = standard_attention_demo(Q, K, V)
_ = linear_attention_demo(Q, K, V)