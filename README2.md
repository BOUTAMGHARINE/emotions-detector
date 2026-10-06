# CNN et VGG16 : comprendre la convolution, les filtres, ReLU et le pooling

## 1. Vue d'ensemble

Un CNN (Convolutional Neural Network) traite une image progressivement afin d'extraire des caractéristiques de plus en plus complexes.

Le principe général est :

```text
Image
  ↓
Convolution
  ↓
Feature Maps
  ↓
ReLU
  ↓
Pooling
  ↓
Nouvelles convolutions
  ↓
Feature Maps plus abstraites
  ↓
Classification
```

Dans un réseau comme VGG16, les premières couches détectent des éléments simples comme les contours et les lignes, tandis que les couches profondes apprennent des formes et des représentations plus complexes.

---

# 2. Une image est une matrice de pixels

Une image peut être représentée comme une matrice de nombres.

Pour une image grayscale :

```text
5 × 5

[ 1  1  1  1  1
  1  1  8  1  1
  1  1  8  1  1
  1  1  8  1  1
  1  1  1  1  1 ]
```

Chaque nombre représente l'intensité d'un pixel.

Pour une image RGB, chaque pixel possède trois valeurs :

```text
R = Red
G = Green
B = Blue
```

Une image RGB peut donc être représentée par :

```text
Height × Width × Channels
```

Par exemple :

```text
224 × 224 × 3
```

signifie :

* hauteur = 224 pixels
* largeur = 224 pixels
* 3 canaux = RGB

---

# 3. La convolution

La convolution permet à un CNN de détecter des caractéristiques dans une image.

Elle utilise un petit filtre, aussi appelé :

* kernel
* filtre
* convolutional filter

Exemple d'un filtre 3×3 :

```text
[ -1   2  -1
  -1   2  -1
  -1   2  -1 ]
```

Le filtre est déplacé sur différentes régions de l'image.

À chaque position :

1. On prend une région de l'image de même taille que le filtre.
2. On multiplie les valeurs pixel par pixel.
3. On additionne les résultats.
4. On obtient une seule valeur.

Exemple :

```text
Image patch:

[ 1  1  1
  1  1  8
  1  1  8 ]

Filtre:

[ -1  2  -1
  -1  2  -1
  -1  2  -1 ]
```

On effectue :

```text
1×(-1) + 1×2 + 1×(-1)
+ 1×(-1) + 1×2 + 8×(-1)
+ 1×(-1) + 1×2 + 8×(-1)
```

Le résultat obtenu est une seule valeur.

Cette opération est répétée à différentes positions.

---

# 4. Pourquoi un filtre produit une Feature Map ?

Un filtre ne regarde pas seulement une partie de l'image.

Il se déplace sur toute l'image :

```text
Image
 ↓
[patch] → valeur
   ↓
[patch] → valeur
   ↓
[patch] → valeur
   ↓
...
```

Chaque position produit une valeur.

Toutes ces valeurs sont regroupées dans une matrice :

```text
Feature Map
```

Donc :

```text
1 filtre
    ↓
scan de toute l'image
    ↓
1 Feature Map
```

Une Feature Map indique approximativement où le motif recherché par le filtre est présent.

---

# 5. Les filtres apprennent des caractéristiques

Dans un vrai CNN, les filtres ne sont généralement pas choisis manuellement.

Le réseau apprend leurs poids pendant l'entraînement.

Un filtre peut progressivement apprendre à détecter :

* des lignes verticales
* des lignes horizontales
* des contours
* des textures
* des formes
* des parties d'objets

Par exemple :

```text
Premières couches
→ contours
→ lignes
→ textures

Couches intermédiaires
→ formes
→ parties d'objets

Couches profondes
→ représentations complexes
→ caractéristiques sémantiques
```

Il ne faut cependant pas imaginer qu'un filtre est explicitement programmé pour "chercher une queue de chien". Le réseau apprend lui-même les caractéristiques utiles à partir des données.

---

# 6. Plusieurs filtres → plusieurs Feature Maps

Supposons qu'une couche convolutionnelle possède 128 filtres.

On obtient :

```text
Filter 1   → Feature Map 1
Filter 2   → Feature Map 2
Filter 3   → Feature Map 3
...
Filter 128 → Feature Map 128
```

Donc :

```text
128 filtres
    ↓
128 Feature Maps
```

Si chaque Feature Map possède une taille de :

```text
56 × 56
```

alors la sortie est :

```text
56 × 56 × 128
```

Cela signifie exactement :

```text
128 matrices
de taille
56 × 56
```

On peut imaginer :

```text
Feature Map 1   → 56 × 56
Feature Map 2   → 56 × 56
Feature Map 3   → 56 × 56
...
Feature Map 128 → 56 × 56
```

---

# 7. Convolution et nombre de canaux

La convolution peut conserver ou modifier les dimensions spatiales selon ses paramètres.

Dans VGG, les convolutions utilisent généralement :

```text
Filter = 3 × 3
Stride = 1
Padding = 1
```

Avec ces paramètres, la convolution conserve la hauteur et la largeur.

Par exemple :

```text
56 × 56 × 64
      ↓
Conv 3×3, 128 filtres
      ↓
56 × 56 × 128
```

La hauteur et la largeur restent :

```text
56 × 56
```

Mais le nombre de Feature Maps passe de :

```text
64 → 128
```

Pourquoi ?

Parce que la nouvelle couche possède 128 filtres.

---

# 8. Pourquoi 5×5 devient 3×3 ?

Considérons :

```text
Image = 5 × 5
Filter = 3 × 3
Stride = 1
Padding = 0
```

La formule générale est :

```text
Output = floor((N + 2P - F) / S) + 1
```

Avec :

```text
N = 5
F = 3
P = 0
S = 1
```

On obtient :

```text
Output = (5 + 0 - 3) / 1 + 1
       = 3
```

Donc :

```text
5 × 5
  ↓
3 × 3 filter
  ↓
3 × 3 output
```

Sans padding, le filtre ne peut pas dépasser les limites de l'image.

---

# 9. Le Padding

Le padding consiste à ajouter une bordure autour de l'image, généralement remplie de zéros.

Exemple :

```text
Image :

[ 1  1  1
  1  8  1
  1  1  1 ]
```

Avec padding :

```text
[ 0  0  0  0  0
  0  1  1  1  0
  0  1  8  1  0
  0  1  1  1  0
  0  0  0  0  0 ]
```

Le padding permet notamment au filtre de mieux traiter les pixels situés sur les bords.

Dans VGG :

```text
3×3 convolution
stride = 1
padding = 1
```

permet de conserver les dimensions spatiales.

Exemple :

```text
56 × 56
   ↓
Conv 3×3, padding 1
   ↓
56 × 56
```

Ainsi, dans VGG, ce n'est généralement pas la convolution qui réduit la taille spatiale.

C'est principalement le pooling qui le fait.

---

# 10. Le ReLU

Après la convolution, on applique généralement une fonction d'activation appelée ReLU.

Formule :

```text
ReLU(x) = max(0, x)
```

Exemple :

```text
Avant ReLU :

[ -2   3  -1
   4  -5   2
  -3   1   6 ]
```

Après ReLU :

```text
[ 0   3   0
  4   0   2
  0   1   6 ]
```

Les valeurs positives restent identiques.

Les valeurs négatives deviennent zéro.

---

# 11. Pourquoi utiliser ReLU ?

ReLU sert principalement à introduire de la non-linéarité dans le réseau.

Sans fonction d'activation, plusieurs opérations linéaires successives peuvent être équivalentes à une seule transformation linéaire.

Le réseau serait alors beaucoup plus limité dans les fonctions complexes qu'il pourrait apprendre.

ReLU permet donc au CNN d'apprendre des relations non linéaires.

On peut résumer :

```text
Convolution
→ extrait des caractéristiques

ReLU
→ introduit la non-linéarité
```

ReLU n'est donc pas un filtre de détection.

Les filtres apprennent des caractéristiques.

ReLU transforme les activations produites par ces filtres.

---

# 12. Où ReLU est-elle appliquée ?

ReLU est appliquée aux valeurs produites par la convolution.

Par exemple :

```text
Image
  ↓
Convolution
  ↓
Feature Map
  ↓
ReLU
  ↓
Feature Map activée
```

Chaque valeur de la Feature Map passe par ReLU.

Exemple :

```text
Feature Map :

[ -2   5
   3  -1 ]

       ↓ ReLU

[  0   5
   3   0 ]
```

---

# 13. Le Pooling

Le pooling réduit la résolution spatiale des Feature Maps.

Dans VGG, on utilise principalement :

```text
Max Pooling
```

avec :

```text
2 × 2
Stride = 2
```

Le Max Pooling prend la plus grande valeur d'une petite région.

Exemple :

```text
Feature Map :

[ 0  2  0  5
  1  8  3  2
  0  4  6  1
  7  2  3  9 ]
```

On divise en régions 2×2 :

```text
[ 0  2 ]   [ 0  5 ]
[ 1  8 ]   [ 3  2 ]
```

On prend le maximum :

```text
8    5
```

Pour la deuxième partie :

```text
[ 0  4 ]   [ 6  1 ]
[ 7  2 ]   [ 3  9 ]
```

On obtient :

```text
7    9
```

Résultat :

```text
[ 8  5
  7  9 ]
```

---

# 14. Le pooling ne travaille pas directement sur l'image originale

Le pooling travaille généralement sur les Feature Maps produites par les convolutions.

Le processus est donc :

```text
Image
  ↓
Convolution
  ↓
Feature Maps
  ↓
ReLU
  ↓
Pooling
  ↓
Feature Maps plus petites
```

Si une convolution produit :

```text
56 × 56 × 128
```

le pooling peut produire :

```text
28 × 28 × 128
```

Le nombre de Feature Maps reste généralement le même.

Ce qui diminue est la résolution spatiale.

---

# 15. Convolution vs ReLU vs Pooling

| Opération       | Fonction                         | Paramètres appris ? |
| --------------- | -------------------------------- | ------------------- |
| Convolution     | Extraire des caractéristiques    | Oui                 |
| ReLU            | Introduire de la non-linéarité   | Non                 |
| Max Pooling     | Réduire la résolution            | Non                 |
| Fully Connected | Classification / décision finale | Oui                 |

Une distinction importante :

```text
Convolution → apprend des filtres
ReLU        → fonction fixe
Pooling     → opération fixe
```

---

# 16. Pourquoi réduire la résolution ?

À mesure que le réseau devient plus profond :

```text
Hauteur/Largeur ↓
Nombre de Feature Maps ↑
```

Par exemple :

```text
112 × 112 × 64
        ↓
56 × 56 × 128
        ↓
28 × 28 × 256
        ↓
14 × 14 × 512
        ↓
7 × 7 × 512
```

Le réseau perd progressivement des détails spatiaux très précis.

En échange, il construit des représentations plus riches et plus abstraites.

On peut résumer :

```text
Résolution spatiale ↓
Représentation des caractéristiques ↑
```

---

# 17. VGG16

VGG16 est une architecture de CNN développée par le Visual Geometry Group de l'Université d'Oxford.

L'entrée originale utilisée pour ImageNet est :

```text
224 × 224 × 3
```

VGG16 est organisé en 5 blocs convolutionnels.

## Architecture des 5 blocs

```text
Block 1
Conv 3×3, 64
Conv 3×3, 64
Max Pooling
        ↓
112 × 112 × 64
```

```text
Block 2
Conv 3×3, 128
Conv 3×3, 128
Max Pooling
        ↓
56 × 56 × 128
```

```text
Block 3
Conv 3×3, 256
Conv 3×3, 256
Conv 3×3, 256
Max Pooling
        ↓
28 × 28 × 256
```

```text
Block 4
Conv 3×3, 512
Conv 3×3, 512
Conv 3×3, 512
Max Pooling
        ↓
14 × 14 × 512
```

```text
Block 5
Conv 3×3, 512
Conv 3×3, 512
Conv 3×3, 512
Max Pooling
        ↓
7 × 7 × 512
```

---

# 18. VGG16 en résumé

L'architecture complète peut être représentée ainsi :

```text
Input
224 × 224 × 3
        ↓
Block 1
Conv64 → ReLU
Conv64 → ReLU
MaxPool
        ↓
112 × 112 × 64
        ↓
Block 2
Conv128 → ReLU
Conv128 → ReLU
MaxPool
        ↓
56 × 56 × 128
        ↓
Block 3
Conv256 → ReLU
Conv256 → ReLU
Conv256 → ReLU
MaxPool
        ↓
28 × 28 × 256
        ↓
Block 4
Conv512 → ReLU
Conv512 → ReLU
Conv512 → ReLU
MaxPool
        ↓
14 × 14 × 512
        ↓
Block 5
Conv512 → ReLU
Conv512 → ReLU
Conv512 → ReLU
MaxPool
        ↓
7 × 7 × 512
        ↓
Flatten
        ↓
25 088 valeurs
        ↓
Fully Connected 4096
        ↓
Fully Connected 4096
        ↓
Fully Connected 1000
        ↓
Softmax
```

---

# 19. Pourquoi 25 088 valeurs après Flatten ?

Après le dernier pooling :

```text
7 × 7 × 512
```

On transforme cette représentation 3D en un seul vecteur.

Calcul :

```text
7 × 7 × 512
= 49 × 512
= 25 088
```

Donc :

```text
7 × 7 × 512
      ↓
Flatten
      ↓
25 088
```

---

# 20. Pourquoi VGG16 s'appelle "16" ?

VGG16 possède :

```text
13 couches convolutionnelles
+
3 couches fully connected
=
16 couches avec paramètres appris
```

Donc :

```text
13 Conv
+
3 FC
=
16
```

Les couches suivantes ne sont pas comptées dans ce nombre :

```text
ReLU
Max Pooling
```

car elles ne possèdent pas de poids appris.

---

# 21. Évolution des Feature Maps dans VGG16

| Bloc    | Couches Conv | Filtres | Sortie         |
| ------- | -----------: | ------: | -------------- |
| Block 1 |            2 |      64 | 112 × 112 × 64 |
| Block 2 |            2 |     128 | 56 × 56 × 128  |
| Block 3 |            3 |     256 | 28 × 28 × 256  |
| Block 4 |            3 |     512 | 14 × 14 × 512  |
| Block 5 |            3 |     512 | 7 × 7 × 512    |

On remarque deux phénomènes :

```text
Hauteur/Largeur :
224 → 112 → 56 → 28 → 14 → 7

Nombre de Feature Maps :
3 → 64 → 128 → 256 → 512 → 512
```

Le réseau réduit donc progressivement la résolution spatiale tout en augmentant la profondeur de sa représentation.

---

# 22. Que signifie "la profondeur augmente" ?

Quand on dit :

```text
56 × 56 × 128
```

le `128` représente le nombre de canaux ou Feature Maps.

Ce n'est pas la largeur de l'image.

On peut imaginer :

```text
56 × 56 × 128

        ↓

128 Feature Maps

┌──────────────┐
│ 56 × 56      │ → Feature Map 1
├──────────────┤
│ 56 × 56      │ → Feature Map 2
├──────────────┤
│ 56 × 56      │ → Feature Map 3
├──────────────┤
│     ...      │
├──────────────┤
│ 56 × 56      │ → Feature Map 128
└──────────────┘
```

Chaque Feature Map correspond à la réponse d'un filtre différent.

---

# 23. La logique complète d'un CNN

On peut maintenant relier toutes les notions :

```text
IMAGE
224 × 224 × 3
        ↓
CONVOLUTION
plusieurs filtres appris
        ↓
FEATURE MAPS
chaque filtre → une Feature Map
        ↓
RELU
supprime les activations négatives
et introduit la non-linéarité
        ↓
POOLING
réduit la résolution spatiale
        ↓
NOUVELLE CONVOLUTION
nouveaux filtres
        ↓
NOUVELLES FEATURE MAPS
représentations plus complexes
        ↓
...
        ↓
FEATURES PROFONDES
        ↓
FLATTEN
        ↓
FULLY CONNECTED
        ↓
CLASSIFICATION
```

---

# 24. Les points essentiels à retenir

### Convolution

```text
Un filtre parcourt l'image
et produit une Feature Map.
```

### Filtre

```text
Un filtre possède des poids appris
et cherche une caractéristique.
```

### Feature Map

```text
1 filtre → 1 Feature Map
```

Donc :

```text
128 filtres → 128 Feature Maps
```

### ReLU

```text
ReLU(x) = max(0, x)
```

Elle est appliquée aux activations produites par la convolution et introduit la non-linéarité.

### Pooling

```text
Le pooling travaille sur les Feature Maps.
```

Il réduit principalement :

```text
Height × Width
```

mais conserve généralement le nombre de canaux.

### VGG

VGG utilise principalement :

```text
Conv 3×3
Stride = 1
Padding = 1
```

puis :

```text
Max Pooling 2×2
Stride = 2
```

La convolution conserve donc généralement la résolution spatiale, tandis que le pooling la réduit.

### VGG16

```text
13 Conv
+
3 Fully Connected
=
16 couches à paramètres appris
```

---

# 25. Mental model final

Le modèle mental le plus important est :

```text
Image
  ↓
Les filtres cherchent des motifs
  ↓
Chaque filtre produit une Feature Map
  ↓
ReLU ajoute de la non-linéarité
  ↓
Pooling réduit la résolution
  ↓
Les couches suivantes combinent les motifs simples
  ↓
Des représentations de plus en plus complexes apparaissent
  ↓
La partie finale utilise ces représentations
pour effectuer la classification
```

En une phrase :

> Un CNN transforme progressivement une image en une représentation de caractéristiques de plus en plus abstraites, en utilisant des convolutions pour extraire des motifs, ReLU pour introduire la non-linéarité et le pooling pour réduire la résolution spatiale.
