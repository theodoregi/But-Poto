L'objectif de ce projet est de détecter des poteaux (ainsi que les buts que forment ces poteaux) pour des robots jouant au football, à partir des images reçues de leurs caméras.

L'objectif est de modéliser de manière précise, et si possible rapide, les cages.

Le fichier main.py permet de voir rapidement tout ce qu'il est possible de faire avec les programmes implémentés.
Ce script Python effectue les opérations suivantes :\
&nbsp;&nbsp;&nbsp;&nbsp;1- Choix d'une image. \
&nbsp;&nbsp;&nbsp;&nbsp;2- Création d'un masque pour l'image.\
&nbsp;&nbsp;&nbsp;&nbsp;3- Détection des poteaux et du but associé sur l'image.\
&nbsp;&nbsp;&nbsp;&nbsp;4- Benchmarks sur N images (10 par défaut).\
&nbsp;&nbsp;&nbsp;&nbsp;5- Détection manuelle des poteaux et du but associé à partir d'une liste. Cette détection manuelle sert de référence pour le calcul de la précision de la détection basée sur le traitement d'image. Des explications plus complètes sont proposées ci-dessous.\
&nbsp;&nbsp;&nbsp;&nbsp;6- Affichage des statistiques de précision.\

Des tests manuels peuvent être réalisés en suivant les étapes suivantes :\
&nbsp;&nbsp;&nbsp;&nbsp;- Lancez le fichier ./src/register_goals.py dans un environnement Python.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Vous devrez vous-même détecter les cages en disposant 4 points sur l'image pour former un triangle.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Après avoir choisi les 4 points correspondant aux coins des buts, appuyez sur n'importe quelle touche de votre clavier.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Le rectangle formé sera affiché sur l'image. Appuyez de nouveau sur n'importe quelle touche pour continuer.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Il vous sera demandé si vous souhaitez enregistrer cet essai. Tapez 'y' puis <Entrée> dans votre terminal pour valider, 'n' et <Entrée> sinon.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Répétez cela autant de fois que nécessaire pour obtenir un échantillon de tests suffisant pour la comparaison avec la génération du rectangle correspondant aux buts qui vous est proposée.\
&nbsp;&nbsp;&nbsp;&nbsp;- Lancez le fichier ./src/compute_goal_surface.py dans un environnement Python.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Sur l'image, les buts détectés par l'algorithme apparaîtront en bleu, tandis que les buts que vous avez détectés seront en rouge.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Dans le terminal, vous obtiendrez diverses métriques permettant d'évaluer l'algorithme, telles que sa précision par rapport à votre prédiction manuelle, ou encore l'indice de Jaccard associé. L'indice de Jaccard correspond à l'aire de l'intersection des deux rectangles divisée par l'aire de l'union des deux rectangles. En notant A l'aire du rectangle issu de la prédiction algorithmique des buts et M l'aire du rectangle de la détection manuelle, on a :\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Jaccard(A,M) = |A ∩ M| / |A ∪ M|\
