# Comprendre les CNN : convolution et apprentissage

Ce document résume les concepts étudiés pour comprendre le fonctionnement d'un réseau de neurones convolutif (CNN) dans mon projet de détection des émotions à partir d'images faciales.

## 1. La couche Conv2D et les filtres

Dans mon modèle, j'utilise la couche suivante :

```python
layers.Conv2D(
    32,
    (3, 3),
    activation="relu",
    padding="same",
    input_shape=(48, 48, 1)
)
```

Chaque paramètre a un rôle précis :

- `32` : nombre de filtres appris par la couche.
- `(3, 3)` : taille spatiale de chaque filtre.
- `activation="relu"` : applique la fonction d'activation ReLU.
- `padding="same"` : conserve les dimensions spatiales de sortie lorsque le stride est égal à 1.
- `input_shape=(48, 48, 1)` : image de 48 × 48 pixels possédant un seul canal en niveaux de gris.

Un filtre possède 9 poids de convolution pour une image à un seul canal, auxquels s'ajoute un biais.

Le nombre total de paramètres entraînables de cette couche est :

\[
(3 \times 3 \times 1 + 1) \times 32 = 320
\]

La couche possède donc **320 paramètres entraînables**.

## 2. Comment fonctionne un filtre ?

Un filtre est une petite matrice de poids qui parcourt différentes régions de l'image.

Considérons une région de 3 × 3 pixels :

\[
X =
\begin{bmatrix}
10 & 40 & 70\\
20 & 50 & 80\\
30 & 60 & 90
\end{bmatrix}
\]

Le filtre possède les poids suivants :

\[
K =
\begin{bmatrix}
1 & 1 & 1\\
0 & 0 & 0\\
-1 & -1 & -1
\end{bmatrix}
\]

Le filtre multiplie chaque poids par le pixel correspondant, additionne les produits et ajoute un biais \(b\).

Le calcul est :

\[
\begin{aligned}
z ={}&10(1)+40(1)+70(1)\\
&+20(0)+50(0)+80(0)\\
&+30(-1)+60(-1)+90(-1)+b\\
={}&-60+b
\end{aligned}
\]

La fonction ReLU transforme ensuite le résultat :

\[
y=\max(0,z)
\]

Si \(b=0\), alors \(z=-60\), donc \(y=0\).

Ce calcul produit une valeur à une position de sortie. Le filtre répète ensuite le même type de calcul sur d'autres régions de l'image.

## 3. Où sont les neurones artificiels ?

Un neurone artificiel effectue généralement les opérations suivantes :

1. Il reçoit des valeurs d'entrée.
2. Il multiplie ces valeurs par des poids.
3. Il additionne les produits et un biais.
4. Il applique une fonction d'activation.
5. Il produit une valeur de sortie.

Dans une couche convolutive, chaque application du filtre à une position de l'image produit une activation correspondant à la sortie d'un neurone convolutionnel.

Il faut distinguer trois notions.

**Le filtre :** il contient les poids appris et utilise un biais pour effectuer les calculs.

**L'activation :** c'est la valeur produite à une position donnée après le calcul et l'application de la fonction d'activation.

**La carte de caractéristiques (`feature map`) :** c'est l'ensemble des activations produites par le même filtre à différentes positions de l'image.

Un même filtre réutilise ses poids à chaque position. Il ne possède donc pas un ensemble de poids différent pour chaque région de l'image.

## 4. Combien de calculs sont effectués ?

Mon image d'entrée mesure 48 × 48 pixels.

Avec `padding="same"` et un stride de 1, chaque filtre produit une carte de caractéristiques de 48 × 48.

Le nombre d'activations produites par un filtre est :

\[
48 \times 48 = 2304
\]

Avec 32 filtres :

\[
32 \times 2304 = 73728
\]

La couche produit donc **73 728 activations pour une image**.

Ces activations ne représentent pas 73 728 ensembles de poids indépendants. La couche apprend seulement 32 ensembles de poids de convolution, chacun réutilisé à différentes positions de l'image.

## 5. Comment les poids sont-ils appris ?

Au début de l'entraînement, les poids sont initialisés avec des valeurs initiales, généralement pseudo-aléatoires. Le réseau les ajuste pendant l'entraînement pour améliorer ses prédictions.

Le processus comporte quatre étapes principales.

### 1. Prédiction (Forward Pass)

Le réseau reçoit une image et prédit les probabilités associées aux sept émotions.

### 2. Calcul de la perte (Loss)

On mesure l'erreur entre les probabilités prédites et la véritable classe de l'image.

### 3. Rétropropagation (Backpropagation)

On calcule les gradients de la fonction de perte par rapport aux paramètres entraînables du réseau.

### 4. Mise à jour avec Adam

L'optimiseur Adam utilise les gradients pour ajuster les poids et les biais.

Ces étapes se répètent sur les lots d'images (`batches`) pendant l'entraînement.

## 6. Pourquoi la dérivée n'est-elle pas toujours nulle ?

On utilise la fonction suivante comme exemple mathématique :

\[
L(w)=(w-3)^2
\]

Sa dérivée par rapport à \(w\) est :

\[
\frac{dL}{dw}=2(w-3)
\]

Si \(w=1\), alors :

\[
\frac{dL}{dw}=2(1-3)=-4
\]

Même si le poids possède une valeur fixe à cet instant, on peut étudier comment la fonction de perte varierait si ce poids changeait légèrement.

Dans un CNN, la fonction de perte dépend de nombreux poids et biais :

\[
L=L(w_1,w_2,\ldots,w_n)
\]

La rétropropagation calcule les dérivées partielles de la perte par rapport aux paramètres :

\[
\frac{\partial L}{\partial w_i}
\]

Ces gradients indiquent comment la perte varie par rapport à chaque paramètre. L'optimiseur les utilise ensuite pour ajuster les poids et les biais.

## 7. Quelle fonction de perte utilise mon CNN ?

Dans mon code, j'utilise :

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

La fonction `sparse_categorical_crossentropy` est adaptée à la classification multiclasse lorsque les vraies classes sont représentées par des entiers.

Dans mon projet, les sept émotions sont représentées par des labels allant de `0` à `6`.

Pour une image, la fonction de perte s'écrit :

\[
L=-\ln(p_{\text{vraie classe}})
\]

où \(p_{\text{vraie classe}}\) représente la probabilité prédite pour l'émotion correcte.

### Exemple 1 : bonne prédiction

Si l'image représente une personne heureuse et que le réseau attribue à `Happy` une probabilité de 0,8 :

\[
L=-\ln(0.8)\approx 0.223
\]

La perte est relativement faible.

### Exemple 2 : mauvaise prédiction

Si le réseau attribue à `Happy` une probabilité de 0,1 :

\[
L=-\ln(0.1)\approx 2.303
\]

La perte est beaucoup plus élevée.

La fonction pénalise donc davantage les prédictions qui attribuent une faible probabilité à la classe correcte.

Les probabilités prédites dépendent des poids du réseau. La perte dépend donc elle aussi de ces poids :

\[
L(w)=-\ln(p_{\text{vraie classe}}(w))
\]

Cette dépendance permet de calculer les gradients et d'apprendre.

## 8. Les trois paramètres de `model.compile()`

Dans le code suivant :

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

Chaque paramètre a une fonction différente.

| Paramètre | Rôle |
|---|---|
| `optimizer="adam"` | Ajuste les poids et les biais à partir des gradients. |
| `loss="sparse_categorical_crossentropy"` | Mesure l'erreur de classification. |
| `metrics=["accuracy"]` | Mesure la proportion de prédictions correctes. |

Il ne faut pas confondre la perte et l'accuracy :

- **La perte (`loss`)** sert à guider l'apprentissage en fournissant un objectif différentiable pour le calcul des gradients.
- **L'accuracy** mesure la proportion de prédictions correctes et sert à évaluer les performances de classification.

Dans cette configuration, Adam optimise la fonction de perte et non directement l'accuracy.

## 9. Résumé du fonctionnement d'un CNN

Le mécanisme général peut être résumé par cette chaîne :

```text
Image d'entrée (48 × 48 × 1)
          |
          v
     Convolution
          |
          v
  Fonction d'activation ReLU
          |
          v
 Cartes de caractéristiques
          |
          v
      Couches suivantes
          |
          v
 Probabilités des 7 émotions
          |
          v
     Fonction de perte
          |
          v
     Rétropropagation
          |
          v
 Calcul des gradients
          |
          v
 Mise à jour des poids avec Adam
          |
          v
     Nouvelle prédiction
```

Pendant l'entraînement, le réseau répète ce processus pour apprendre des caractéristiques visuelles utiles à la classification des émotions.

## 10. Prochaine étape : comprendre la rétropropagation

La prochaine étape consiste à étudier plus en détail la rétropropagation et la règle de dérivation en chaîne.

L'objectif sera de comprendre comment le gradient de la fonction de perte est transmis depuis la sortie du réseau jusqu'aux filtres de convolution.

Les notions à étudier sont :

- Le calcul de la dérivée de la perte par rapport aux probabilités prédites.
- La propagation des gradients à travers la fonction ReLU.
- Le calcul des gradients par rapport aux poids et aux biais d'une convolution.
- La manière dont Adam utilise les gradients pour mettre à jour les paramètres.

C'est le lien mathématique entre la prédiction finale du CNN et l'apprentissage de ses filtres.


# Comprendre `padding="same"` dans un CNN

## 1. Le rôle du padding

Dans un CNN, une couche `Conv2D` déplace un filtre (kernel) sur une image pour détecter des caractéristiques comme les contours, les textures ou les formes.

Le paramètre `padding="same"` ajoute des pixels autour de l'image, généralement des zéros, afin de conserver les dimensions spatiales de la sortie lorsque le stride est égal à 1.

**Important :** le padding ne sert pas à remplir des trous dans l'image. Il permet au filtre de traiter les bords tout en conservant la taille de la sortie.

## 2. Exemple sans padding

Supposons que nous ayons :
- Une image de `4 × 4` pixels.
- Un filtre de `3 × 3`.
- Un stride de `1`.

La formule de la convolution sans padding est :

\[
N_{\text{sortie}}=\frac{N_{\text{entrée}}-F}{S}+1
\]

Où :
- \(N_{\text{entrée}}\) : taille de l'image.
- \(F\) : taille du filtre.
- \(S\) : stride.

En remplaçant les valeurs :

\[
N_{\text{sortie}}=\frac{4-3}{1}+1=2
\]

La sortie mesure donc :

\[
2\times2
\]

Pourquoi ?

Le filtre doit rester entièrement à l'intérieur de l'image. Il peut se déplacer vers la droite et vers le bas, mais il ne peut pas dépasser les bords.

Il possède donc seulement :
- 2 positions horizontales.
- 2 positions verticales.

Au total, il produit \(2\times2=4\) positions de sortie.

## 3. Pourquoi ajouter une bordure de zéros ?

L'objectif est de conserver une sortie de `4 × 4` au lieu de `2 × 2`.

Pour cela, on ajoute une bordure d'un pixel sur chaque côté de l'image.

La nouvelle taille devient :

\[
4+2P=6
\]

Où \(P\) représente la largeur du padding.

On obtient :

\[
P=\frac{6-4}{2}=1
\]

L'image passe donc de `4 × 4` à `6 × 6`.

La formule de sortie devient :

\[
N_{\text{sortie}}=N+2P-F+1
\]

En remplaçant :

\[
N_{\text{sortie}}=4+2(1)-3+1=4
\]

La sortie mesure maintenant `4 × 4`.

## 4. Formule générale pour `padding="same"`

Pour une convolution avec un stride de 1 et un filtre de taille impaire :

\[
P=\frac{F-1}{2}
\]

Exemples :

| Taille du filtre | Padding de chaque côté |
|---|---:|
| `3 × 3` | 1 pixel |
| `5 × 5` | 2 pixels |
| `7 × 7` | 3 pixels |

Cette formule suppose un stride de 1. Avec un stride supérieur à 1, le calcul de la sortie est différent.

## 5. Le filtre couvre-t-il toute l'image sans padding ?

Il faut distinguer deux notions :

**Couvrir tous les pixels au moins une fois :** c'est possible sans padding. Le filtre peut analyser les pixels de toute l'image.

**Produire une activation pour chaque position spatiale de l'image :** ce n'est pas possible sans padding dans notre exemple `4 × 4` avec un filtre `3 × 3`.

Sans padding, la sortie est `2 × 2`. Avec `padding="same"`, elle est `4 × 4`.

Le padding permet donc de conserver une sortie de même hauteur et largeur que l'image d'entrée.

## 6. Application à mon CNN

Dans mon modèle :

```python
layers.Conv2D(
    32,
    (3, 3),
    activation="relu",
    padding="same",
    input_shape=(48, 48, 1)
)
```

L'image possède les dimensions `48 × 48 × 1` :
- 48 pixels de hauteur.
- 48 pixels de largeur.
- 1 canal, car l'image est en niveaux de gris.

Après la convolution avec `padding="same"` et un stride de 1 :

\[
48\times48\times1
\longrightarrow
48\times48\times32
\]

Les 32 filtres produisent 32 cartes de caractéristiques de taille `48 × 48`.

Ensuite, le modèle applique :

```python
layers.MaxPooling2D((2, 2))
```

Cette couche réduit les dimensions spatiales de moitié :

\[
48\times48\times32
\longrightarrow
24\times24\times32
\]

Le pooling réduit donc la taille des cartes de caractéristiques, tandis que le padding conserve les dimensions pendant la convolution.

## 7. À retenir

- `Conv2D` applique des filtres pour extraire des caractéristiques.
- `padding="same"` ajoute une bordure pour conserver les dimensions spatiales lorsque le stride est de 1.
- Sans padding, le filtre reste à l'intérieur de l'image et la sortie est généralement plus petite.
- Le padding ne restaure pas les dimensions après le pooling.
- `MaxPooling2D((2, 2))` réduit la hauteur et la largeur, généralement de moitié.

**Résumé du fonctionnement :**

\[
48\times48
\xrightarrow{\text{Conv2D, same}}
48\times48
\xrightarrow{\text{MaxPooling}}
24\times24
\]
