Margo Biret, Lucas Gavériaux, Théodore Gigault

# Détection de poteaux de football sur une image

Le but de ce projet consiste à mettre en œuvre la détection de poteaux, ainsi que la reconnaissance des buts formés par ces poteaux, à partir des images capturées par les caméras des robots participant au football.
L'objectif principal est de réaliser une modélisation précise et, dans la mesure du possible, rapide des cages.

## Contexte

Ce projet s'inscrit dans notre cursus en ingénierie informatique, plus particulièrement dans la spécialisation en robotique dispensée par l'ENSEIRB-MATMECA.

L'initiative provient de nos enseignants, qui sont également impliqués dans l'équipe Rhoban participant au concours de robotique RoboCup.
RoboCup est une initiative internationale de recherche scientifique et de compétition qui vise à promouvoir le développement de la robotique et de l'intelligence artificielle. L'objectif ultime de RoboCup est de développer des équipes de robots capables de rivaliser avec des équipes humaines dans le football (soccer).

## Table des matières

- [Description](#contexte)
- [Tests manuels](#tests-manuels)
- [Précautions](#précautions)
- [Prérequis](#prérequis)

## Description

Le fichier **main.py** permet de voir rapidement tout ce qu'il est possible de faire avec les programmes implémentés.
Ce script Python effectue les opérations suivantes :\
&nbsp;&nbsp;&nbsp;&nbsp;1. Choix d'une image. \
&nbsp;&nbsp;&nbsp;&nbsp;2. Création d'un masque pour l'image.\
&nbsp;&nbsp;&nbsp;&nbsp;3. Détection des poteaux et du but associé sur l'image.\
&nbsp;&nbsp;&nbsp;&nbsp;4. Benchmarks sur N images (10 par défaut).\
&nbsp;&nbsp;&nbsp;&nbsp;5. Détection manuelle des poteaux et du but associé à partir d'une liste. Cette détection manuelle sert de référence pour le calcul de la précision de la détection basée sur le traitement d'image. Des explications plus complètes sont proposées ci-dessous.\
&nbsp;&nbsp;&nbsp;&nbsp;6- Affichage des statistiques de précision.\

## Tests manuels

Des tests manuels peuvent être réalisés en suivant les étapes suivantes :\
&nbsp;&nbsp;&nbsp;&nbsp;- Lancez le fichier **./src/register_goals.py** dans un environnement Python.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Vous devrez vous-même détecter les cages en disposant 4 points sur l'image pour former un rectangle.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Après avoir choisi les 4 points correspondant aux coins des buts, appuyez sur n'importe quelle touche de votre clavier.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Le rectangle formé sera affiché sur l'image. Appuyez de nouveau sur n'importe quelle touche pour continuer.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Il vous sera demandé si vous souhaitez enregistrer cet essai. Tapez **'y'** puis **<Entrée>** dans votre terminal pour valider, **'n'** et **<Entrée>** sinon.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Répétez cela autant de fois que nécessaire pour obtenir un échantillon de tests suffisant pour la comparaison avec la génération du rectangle correspondant aux buts qui vous est proposée.\
&nbsp;&nbsp;&nbsp;&nbsp;- Lancez le fichier **./src/compute_goal_surface.py** dans un environnement Python.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Sur l'image, les buts détectés par l'algorithme apparaîtront en bleu, tandis que les buts que vous avez détectés seront en rouge.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Dans le terminal, vous obtiendrez diverses métriques permettant d'évaluer l'algorithme, telles que sa précision par rapport à votre prédiction manuelle, ou encore l'indice de Jaccard associé. L'indice de Jaccard correspond à l'aire de l'intersection des deux rectangles divisée par l'aire de l'union des deux rectangles. En notant A l'aire du rectangle issu de la prédiction algorithmique des buts et M l'aire du rectangle de la détection manuelle, on a :\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Jaccard(A,M) = |A ∩ M| / |A ∪ M|\

## Précautions

**Attention**, lors des affichages avec _OpenCV2_ (images des buts, binarisées ou non), il est déconseillé de fermer la fenêtre en cliquant sur la croix située dans le coin supérieur droit. Il est préférable d'appuyer sur n'importe quelle touche du clavier à la place.

Pour les affichages avec _Matplotlib_ (courbes), il est recommandé de fermer la fenêtre en cliquant sur la croix dans le coin supérieur droit.

## Prérequis

Afin de pouvoir faire fonctionner ce projet sur votre machine, il est nécessaire d'installer **Tensorflow**, NumPy (version >=1.17.3 et <1.25.0) et SciPy (version 1.26.1).
