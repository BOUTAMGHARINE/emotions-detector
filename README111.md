# Comprendre VGG-16

![Architecture VGG-16](vgg-neural-network-architecture.webp)

## 1. Qu'est-ce que VGG ?

**VGG** signifie **Visual Geometry Group**, le laboratoire de recherche de l'Université d'Oxford qui a développé cette architecture.

VGG est un **réseau neuronal convolutif (CNN)** utilisé principalement pour la **reconnaissance et la classification d'images**.

Les deux architectures principales sont :

* **VGG-16** : 16 couches avec des paramètres entraînables
* **VGG-19** : 19 couches avec des paramètres entraînables

L'idée principale de VGG est d'utiliser plusieurs petites convolutions **3×3** empilées les unes après les autres.

---

## 2. L'image d'entrée

VGG-16 reçoit généralement une image de taille :

```text
224 × 224 × 3
```

Cela signifie :

* `224` : hauteur
* `224` : largeur
* `3` : canaux de couleur **RGB**

Chaque pixel possède donc trois valeurs :

```text
Pixel = [R, G, B]
```

Par exemple :

```text
[255, 0, 0] → rouge
[0, 255, 0] → vert
[0, 0, 255] → bleu
```

On peut donc considérer une image RGB comme une matrice :

```text
Hauteur × Largeur × Channels
```

---

## 3. Que fait une convolution ?

Une convolution utilise un **filtre**, aussi appelé **kernel**.

Dans VGG, les convolutions utilisent principalement des filtres :

```text
3 × 3
```

Le filtre se déplace sur différentes régions de l'image et effectue des calculs.

```text
Image
┌─────────────────────┐
│                     │
│      ┌─────┐        │
│      │ 3×3 │        │ ← filtre
│      └─────┘        │
│                     │
└─────────────────────┘
```

À chaque position, le filtre produit une valeur indiquant à quel point le motif qu'il a appris est présent à cet endroit.

Le résultat est appelé une **feature map**.

---

## 4. Est-ce qu'un filtre cherche une caractéristique spécifique ?

**Oui, mais progressivement.**

Un filtre apprend automatiquement à détecter certains motifs.

Dans les premières couches, les motifs sont généralement simples :

```text
Contours
Lignes
Orientations
Textures simples
```

Dans les couches plus profondes, les représentations deviennent plus complexes :

```text
Textures
      ↓
Formes
      ↓
Parties d'objets
      ↓
Structures complexes
      ↓
Objet / catégorie
```

Par exemple, pour une image contenant un chien :

```text
Premières couches
→ contours et lignes

Couches intermédiaires
→ textures et formes

Couches profondes
→ parties pouvant correspondre à une oreille,
  un museau, une patte, une queue, etc.

Couches finales
→ caractéristiques permettant de reconnaître
  une catégorie comme "chien"
```

Il faut cependant éviter de penser :

```text
Filtre 1 = queue de chien
Filtre 2 = oreille de chien
```

Le réseau n'est pas programmé de cette manière.

Les filtres **apprennent leurs paramètres pendant l'entraînement** et les représentations deviennent progressivement plus complexes grâce à la combinaison des couches.

---

## 5. Les feature maps

Supposons qu'une couche possède **64 filtres**.

Chaque filtre produit une feature map.

On obtient donc :

```text
224 × 224 × 3
        ↓
64 filtres
        ↓
224 × 224 × 64
```

Le `64` ne représente pas 64 couleurs.

Il représente :

```text
64 feature maps
```

Chaque feature map représente la réponse d'un filtre différent.

On peut donc simplifier :

```text
Image
  ↓
Filtre 1 → Feature map 1
Filtre 2 → Feature map 2
Filtre 3 → Feature map 3
...
Filtre 64 → Feature map 64
```

---

## 6. Max Pooling

Après plusieurs convolutions, VGG utilise du **Max Pooling**.

Exemple :

```text
224 × 224 × 64
        ↓
   Max Pooling
        ↓
112 × 112 × 64
```

Le pooling réduit la dimension spatiale.

Par exemple :

```text
┌───┬───┐
│ 1 │ 5 │
├───┼───┤
│ 2 │ 3 │
└───┴───┘
```

Avec un Max Pooling, on conserve :

```text
5
```

Le pooling permet donc de réduire la taille des représentations tout en conservant les activations importantes.

---

## 7. L'évolution des dimensions

Une caractéristique importante de VGG est que :

**la résolution spatiale diminue progressivement tandis que le nombre de feature maps augmente.**

On peut résumer l'architecture ainsi :

```text
224 × 224 × 3
        ↓
224 × 224 × 64
        ↓
112 × 112 × 64
        ↓
112 × 112 × 128
        ↓
56 × 56 × 128
        ↓
56 × 56 × 256
        ↓
28 × 28 × 256
        ↓
28 × 28 × 512
        ↓
14 × 14 × 512
        ↓
14 × 14 × 512
        ↓
7 × 7 × 512
```

On observe deux choses :

```text
Dimension spatiale :
224 → 112 → 56 → 28 → 14 → 7

Nombre de feature maps :
64 → 128 → 256 → 512
```

Donc plus on avance dans le réseau :

* l'image devient spatialement plus petite ;
* le nombre de caractéristiques augmente ;
* les caractéristiques deviennent progressivement plus abstraites.

---

## 8. Les 3×3 convolutions

L'une des idées importantes de VGG est l'utilisation répétée de petites convolutions :

```text
3 × 3
```

plutôt que de grandes convolutions.

Par exemple :

```text
3×3
 ↓
3×3
 ↓
3×3
```

Cela permet au réseau d'empiler plusieurs transformations et d'apprendre progressivement des caractéristiques plus complexes.

L'idée principale de VGG peut donc être résumée ainsi :

```text
Petites convolutions
        ↓
Plusieurs couches
        ↓
Caractéristiques de plus en plus complexes
```

---

## 9. La partie convolutionnelle

La première grande partie du réseau sert principalement à **extraire les caractéristiques de l'image**.

```text
Image
  ↓
Convolution
  ↓
ReLU
  ↓
Convolution
  ↓
ReLU
  ↓
Max Pooling
  ↓
Convolution
  ↓
...
  ↓
7 × 7 × 512
```

On peut donc considérer cette partie comme un **extracteur de caractéristiques**.

---

## 10. Que signifie 7 × 7 × 512 ?

À la fin de la partie convolutionnelle, on obtient :

```text
7 × 7 × 512
```

Cela signifie :

```text
Hauteur = 7
Largeur = 7
Feature maps = 512
```

Le nombre total de valeurs est :

```text
7 × 7 × 512 = 25 088
```

Ces valeurs peuvent ensuite être transformées en un vecteur :

```text
7 × 7 × 512
       ↓
    Flatten
       ↓
25 088 valeurs
```

---

## 11. Les couches Fully Connected

Après les convolutions vient la partie **Fully Connected**.

Dans VGG-16, on trouve :

```text
25 088
   ↓
4096
   ↓
4096
   ↓
1000
```

Les couches fully connected utilisent les caractéristiques extraites précédemment pour effectuer la classification.

On peut donc résumer :

```text
Image
  ↓
Convolution
  ↓
Extraction de caractéristiques
  ↓
Fully Connected
  ↓
Classification
```

---

## 12. La dernière couche : 1000 classes

Le VGG original a été entraîné sur **ImageNet** avec 1000 catégories.

La dernière couche produit donc 1000 scores.

Par exemple :

```text
chien      → 0.81
chat       → 0.06
voiture    → 0.04
bus        → 0.02
...
```

La fonction **Softmax** transforme les scores en probabilités.

Le réseau peut alors choisir la classe ayant la probabilité la plus élevée.

---

## 13. Pourquoi VGG-16 ?

Le nombre `16` correspond aux **16 couches possédant des paramètres entraînables** :

```text
13 couches convolutionnelles
+
3 couches fully connected
=
16 couches
```

Les couches comme :

```text
Max Pooling
ReLU
Softmax
```

ne sont pas comptées dans ce nombre de 16.

Donc :

```text
VGG-16

13 × Convolution
+
3 × Fully Connected
=
16
```

---

## 14. Vue globale

On peut retenir VGG-16 avec ce schéma :

```text
IMAGE
224 × 224 × 3
      ↓
CONVOLUTIONS 3×3
      ↓
Détection de motifs simples
      ↓
MAX POOLING
      ↓
Réduction de la résolution
      ↓
CONVOLUTIONS
      ↓
Caractéristiques plus complexes
      ↓
MAX POOLING
      ↓
...
      ↓
7 × 7 × 512
      ↓
FLATTEN
      ↓
25 088 valeurs
      ↓
FULLY CONNECTED
      ↓
4096
      ↓
FULLY CONNECTED
      ↓
4096
      ↓
1000 classes
      ↓
SOFTMAX
      ↓
Classification
```

---

## 15. L'idée fondamentale à retenir

Le fonctionnement d'un CNN comme VGG peut être résumé en une phrase :

> **Les premières couches détectent des motifs simples, les couches suivantes combinent ces motifs pour représenter des structures plus complexes, et les dernières couches utilisent ces caractéristiques pour classifier l'image.**

Pour une image de chien, par exemple :

```text
Pixels
  ↓
Contours
  ↓
Textures
  ↓
Formes
  ↓
Parties de l'objet
  ↓
Caractéristiques complexes
  ↓
Chien
```

Le point essentiel est de distinguer :

```text
Dimension spatiale
224 × 224 → 112 × 112 → 56 × 56 → ...

Nombre de caractéristiques
64 → 128 → 256 → 512
```

La première diminue tandis que la seconde augmente.

C'est l'un des principes fondamentaux des architectures CNN.

---

## 16. Résumé très court

```text
VGG-16
│
├── Image : 224 × 224 × 3
│
├── Convolutions 3×3
│   └── Détection de caractéristiques
│
├── Max Pooling
│   └── Réduction de la résolution
│
├── Feature maps
│   └── 64 → 128 → 256 → 512
│
├── 7 × 7 × 512
│
├── Flatten
│   └── 25 088 valeurs
│
├── Fully Connected
│   └── 4096
│
├── Fully Connected
│   └── 4096
│
├── 1000 classes
│
└── Softmax
    └── Probabilités des classes
```

### À retenir

**Un filtre apprend à détecter un motif.**

**Une feature map montre où ce motif est détecté.**

**Plusieurs couches combinent progressivement les motifs simples en caractéristiques plus complexes.**

**Le pooling réduit la résolution spatiale.**

**Le nombre de feature maps augmente au fur et à mesure que le réseau devient profond.**

**VGG-16 possède 13 couches convolutionnelles + 3 couches fully connected = 16 couches à paramètres entraînables.**
