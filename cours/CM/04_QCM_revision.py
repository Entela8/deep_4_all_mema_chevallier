import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    def check_answer(question, correct_answer, explanation):
        """Affiche le feedback apres selection d'une reponse."""
        if question.value is None:
            return mo.md("")
        if question.value == correct_answer:
            return mo.callout(
                mo.md(f"**Correct !**\n\n{explanation}"),
                kind="success",
            )
        return mo.callout(
            mo.md(
                f"**Incorrect.** Bonne reponse : **{correct_answer}**\n\n"
                f"{explanation}"
            ),
            kind="danger",
        )
    return (check_answer,)


@app.cell
def _(mo):
    mo.md(r"""
    # QCM de Revision — Word Embeddings, RNN & LSTM

    Bienvenue dans ce quiz interactif ! L'objectif est de **reactiver vos connaissances** sur les concepts vus en cours avant d'aller plus loin.

    **Regles du jeu :**
    - Repondez a chaque question en selectionnant une reponse
    - Le feedback s'affiche immediatement apres votre selection
    - Le score global est disponible en bas de page

    ---

    ## Partie 1 : Word Embeddings
    """)
    return


# ============================================================
# QUESTION 1 — reponse C
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""
    **Q1.** Qu'est-ce qu'un *word embedding* ?

    - **A.** Un dictionnaire qui associe chaque mot a sa definition
    - **B.** Une matrice de co-occurrence des mots
    - **C.** Une representation vectorielle dense capturant des relations semantiques
    - **D.** Un encodage one-hot de grande dimension
    """)
    return


@app.cell
def _(mo):
    q1 = mo.ui.radio(
        options=["A", "B", "C", "D"],
        label="Votre reponse :",
    )
    q1
    return (q1,)


@app.cell
def _(check_answer, q1):
    check_answer(
        q1,
        "C",
        "Un word embedding est une **representation vectorielle dense** apprise a partir d'un corpus. Contrairement au one-hot (creux, haute dimension), il capture des **relations semantiques** : des mots proches dans l'espace vectoriel ont des sens similaires.",
    )
    return


# ============================================================
# QUESTION 2 — reponse D
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""
    **Q2.** Quel est le role d'une couche d'embedding (`nn.Embedding`) ?

    - **A.** Elle normalise les entrees entre 0 et 1
    - **B.** Elle applique une convolution sur le texte
    - **C.** Elle calcule la frequence de chaque mot dans le corpus
    - **D.** Elle agit comme une table de correspondance : chaque index est associe a un vecteur dense
    """)
    return


@app.cell
def _(mo):
    q2 = mo.ui.radio(
        options=["A", "B", "C", "D"],
        label="Votre reponse :",
    )
    q2
    return (q2,)


@app.cell
def _(check_answer, q2):
    check_answer(
        q2,
        "D",
        "`nn.Embedding` est une **table de lookup** : elle associe chaque index entier a un vecteur dense appris durant l'entrainement. C'est equivalent a une multiplication par une matrice one-hot, mais bien plus efficace.",
    )
    return


# ============================================================
# QUESTION 4 — reponse A
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""
    **Q3.** Que mesure la *similarite cosinus* entre deux embeddings ?

    - **A.** Le cosinus de l'angle entre deux vecteurs, mesurant leur proximite directionnelle
    - **B.** La difference de norme entre les deux vecteurs
    - **C.** La distance euclidienne entre les deux vecteurs
    - **D.** Le produit scalaire brut des deux vecteurs
    """)
    return


@app.cell
def _(mo):
    q4 = mo.ui.radio(
        options=["A", "B", "C", "D"],
        label="Votre reponse :",
    )
    q4
    return (q4,)


@app.cell
def _(check_answer, q4):
    check_answer(
        q4,
        "A",
        "La similarite cosinus mesure le **cosinus de l'angle** entre deux vecteurs. Elle vaut 1 si les vecteurs sont paralleles (meme direction), 0 si orthogonaux, -1 si opposes. Elle ignore la norme et se concentre sur la **direction**.",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Partie 2 : RNN (Reseaux de Neurones Recurrents)
    """)
    return


# ============================================================
# QUESTION 5 — reponse C
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""
    **Q4.** Quelle est la formule de mise a jour de l'etat cache d'un RNN simple ?

    - **A.** $h_t = \sigma(W_{xh} \cdot x_t + b)$
    - **B.** $h_t = \text{ReLU}(W \cdot x_t)$
    - **C.** $h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$
    - **D.** $h_t = W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1}$
    """)
    return


@app.cell
def _(mo):
    q5 = mo.ui.radio(
        options=["A", "B", "C", "D"],
        label="Votre reponse :",
    )
    q5
    return (q5,)


@app.cell
def _(check_answer, q5):
    check_answer(
        q5,
        "C",
        r"La formule du RNN est $h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$. Le point cle est la **recurrence** : l'etat $h_{t-1}$ est reinjecte a chaque pas, ce qui cree la memoire.",
    )
    return


# ============================================================
# QUESTION 6 — reponse D
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""
    **Q5.** Pourquoi les RNN simples souffrent-ils du *vanishing gradient* ?

    - **A.** La fonction tanh sature toujours a 1
    - **B.** Le learning rate est trop grand pour les longues sequences
    - **C.** Le reseau a trop de parametres et overfitte
    - **D.** Le gradient est multiplie par $W_{hh}$ a chaque pas, ce qui le fait tendre vers 0 ou l'infini
    """)
    return


@app.cell
def _(mo):
    q6 = mo.ui.radio(
        options=["A", "B", "C", "D"],
        label="Votre reponse :",
    )
    q6
    return (q6,)


@app.cell
def _(check_answer, q6):
    check_answer(
        q6,
        "D",
        r"Lors de la backpropagation, le gradient est multiplie par $W_{hh}$ a **chaque pas de temps**. Si $\|W_{hh}\| < 1$, le gradient tend vers 0 exponentiellement. Apres 50 pas avec $\|W\| = 0.9$ : $0.9^{50} \approx 0.005$.",
    )
    return


# ============================================================
# QUESTION 7 — reponse A
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""
    **Q6.** Quelles sont les deux limitations majeures des reseaux feedforward par rapport aux donnees sequentielles ?

    - **A.** Ils ont une entree de taille fixe et pas de memoire entre les entrees
    - **B.** Ils ne supportent pas la backpropagation
    - **C.** Ils necessitent toujours un GPU
    - **D.** Ils ne peuvent traiter que du texte
    """)
    return


@app.cell
def _(mo):
    q7 = mo.ui.radio(
        options=["A", "B", "C", "D"],
        label="Votre reponse :",
    )
    q7
    return (q7,)


@app.cell
def _(check_answer, q7):
    check_answer(
        q7,
        "A",
        "Les reseaux feedforward exigent une **entree de taille fixe** et traitent chaque entree **independamment** (pas de memoire). Les RNN resolvent ces deux problemes grace a l'etat cache.",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Partie 3 : LSTM & GRU
    """)
    return


# ============================================================
# QUESTION 8 — reponse B
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""
    **Q7.** Quelles sont les trois portes (gates) d'un LSTM ?

    - **A.** Read gate, Write gate, Erase gate
    - **B.** Forget gate, Input gate, Output gate
    - **C.** Attention gate, Memory gate, Output gate
    - **D.** Reset gate, Update gate, Output gate
    """)
    return


@app.cell
def _(mo):
    q8 = mo.ui.radio(
        options=["A", "B", "C", "D"],
        label="Votre reponse :",
    )
    q8
    return (q8,)


@app.cell
def _(check_answer, q8):
    check_answer(
        q8,
        "B",
        "Le LSTM possede 3 portes : **Forget** (que oublier de $C_{t-1}$), **Input** (que ajouter a $C_t$), **Output** (que produire dans $h_t$). Le GRU, lui, n'a que 2 portes (reset et update).",
    )
    return


# ============================================================
# QUESTION 9 — reponse D
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""
    **Q8.** Quelle est la formule de mise a jour de l'etat de cellule $C_t$ dans un LSTM ?

    - **A.** $C_t = C_{t-1} + x_t$
    - **B.** $C_t = \tanh(W \cdot C_{t-1} + b)$
    - **C.** $C_t = \sigma(W_C \cdot [h_{t-1}, x_t])$
    - **D.** $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
    """)
    return


@app.cell
def _(mo):
    q9 = mo.ui.radio(
        options=["A", "B", "C", "D"],
        label="Votre reponse :",
    )
    q9
    return (q9,)


@app.cell
def _(check_answer, q9):
    check_answer(
        q9,
        "D",
        r"$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ : la memoire est mise a jour par une **combinaison lineaire** entre l'ancien etat (filtre par la forget gate) et les nouvelles informations (filtrees par l'input gate).",
    )
    return


# ============================================================
# QUESTION 10 — reponse A
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""
    **Q9.** Pourquoi le LSTM resout-il le probleme du vanishing gradient ?

    - **A.** Grace a l'addition lineaire dans $C_t$ : si $f_t \approx 1$, le gradient traverse sans modification
    - **B.** Grace a la normalisation automatique des poids a chaque pas de temps
    - **C.** Grace a la multiplication matricielle repetee qui stabilise le gradient
    - **D.** Grace a un learning rate adaptatif integre dans l'architecture
    """)
    return


@app.cell
def _(mo):
    q10 = mo.ui.radio(
        options=["A", "B", "C", "D"],
        label="Votre reponse :",
    )
    q10
    return (q10,)


@app.cell
def _(check_answer, q10):
    check_answer(
        q10,
        "A",
        r"Le secret est l'**addition lineaire** dans $C_t = f_t \odot C_{t-1} + \dots$. La derivee $\partial C_t / \partial C_{t-1} = f_t$. Si $f_t \approx 1$, le gradient traverse **sans attenuation** : c'est l'autoroute du gradient (Constant Error Carousel).",
    )
    return


# ============================================================
# QUESTION 11 — reponse C
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""
    **Q10.** Quelle est la difference principale entre GRU et LSTM ?

    - **A.** Le GRU est plus ancien que le LSTM
    - **B.** Le GRU ne resout pas le vanishing gradient contrairement au LSTM
    - **C.** Le GRU a 2 portes (reset, update) et 1 seul etat, le LSTM a 3 portes et 2 etats ($h$ et $C$)
    - **D.** Le GRU a 3 portes et 2 etats, le LSTM a 2 portes et 1 etat
    """)
    return


@app.cell
def _(mo):
    q11 = mo.ui.radio(
        options=["A", "B", "C", "D"],
        label="Votre reponse :",
    )
    q11
    return (q11,)


@app.cell
def _(check_answer, q11):
    check_answer(
        q11,
        "C",
        "Le GRU (Cho et al., 2014) simplifie le LSTM : **2 portes** (reset $r_t$, update $z_t$) au lieu de 3, et **1 seul etat** $h_t$ (pas de $C_t$ separe). Il a ~25% moins de parametres ($3h^2$ vs $4h^2$).",
    )
    return


# ============================================================
# QUESTION 12 — reponse B
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""
    **Q11.** Quel est le role de la *forget gate* ($f_t$) dans un LSTM ?

    - **A.** Elle normalise l'etat de cellule entre -1 et 1
    - **B.** Elle controle quelle partie de la memoire precedente $C_{t-1}$ est conservee ou oubliee
    - **C.** Elle produit la sortie finale du LSTM
    - **D.** Elle decide quelles nouvelles informations stocker dans la memoire
    """)
    return


@app.cell
def _(mo):
    q12 = mo.ui.radio(
        options=["A", "B", "C", "D"],
        label="Votre reponse :",
    )
    q12
    return (q12,)


@app.cell
def _(check_answer, q12):
    check_answer(
        q12,
        "B",
        r"La forget gate $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ produit des valeurs entre 0 et 1 pour **chaque dimension** de $C_{t-1}$. Une valeur proche de 0 = oublier, proche de 1 = conserver.",
    )
    return


# ============================================================
# SCORE GLOBAL
# ============================================================
@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Score global
    """)
    return


@app.cell
def _(mo, q1, q2, q4, q5, q6, q7, q8, q9, q10, q11, q12):
    _correct = {
        "Q1": ("C", q1.value),
        "Q2": ("D", q2.value),
        "Q3": ("A", q4.value),
        "Q4": ("C", q5.value),
        "Q5": ("D", q6.value),
        "Q6": ("A", q7.value),
        "Q7": ("B", q8.value),
        "Q8": ("D", q9.value),
        "Q9": ("A", q10.value),
        "Q10": ("C", q11.value),
        "Q11": ("B", q12.value),
    }

    _answered = sum(1 for _, ans in _correct.values() if ans is not None)
    _score = sum(1 for correct_ans, ans in _correct.values() if ans == correct_ans)
    _total = len(_correct)

    if _answered == 0:
        mo.md("*Repondez aux questions ci-dessus pour voir votre score.*")
    else:
        _pct = _score / _total * 100
        if _pct == 100:
            _verdict = "Parfait ! Vous maitrisez tous les concepts."
        elif _pct >= 75:
            _verdict = "Tres bien ! Quelques points a revoir."
        elif _pct >= 50:
            _verdict = "Correct, mais plusieurs notions meritent d'etre approfondies."
        else:
            _verdict = "Il serait utile de revoir les cours sur les embeddings et les RNN/LSTM."

        mo.callout(
            mo.md(f"## {_score} / {_total} ({_pct:.0f}%) — {_answered} question(s) repondue(s)\n\n{_verdict}"),
            kind="success" if _pct >= 75 else ("warn" if _pct >= 50 else "danger"),
        )
    return


if __name__ == "__main__":
    app.run()
