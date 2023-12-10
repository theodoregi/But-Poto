L'objectif de ce projet est de détecter des poteaux (ainsi que les buts que forment ces poteaux) pour des robots jouant au football, à partir des images reçues de leurs caméras.

L'objectif est de modéliser de manière précise, et si possible rapide, les cages.

Un ensemble de tests participatifs peut être réalisé selon les étapes suivantes :

    - Lancez le fichier ./src/register_goals.py dans un environnement Python.
        - Vous devrez vous-même détecter les cages en disposant 4 points sur l'image pour former un triangle.
        - Après avoir choisi les 4 points correspondant aux coins des buts, appuyez sur n'importe quelle touche de votre clavier.
        - Le rectangle formé sera affiché sur l'image. Appuyez de nouveau sur n'importe quelle touche pour continuer.
        - Il vous sera demandé si vous souhaitez enregistrer cet essai. Tapez 'y' puis <Entrée> dans votre terminal pour valider, 'n' et <Entrée> sinon.
        - Répétez cela autant de fois que nécessaire pour obtenir un échantillon de tests suffisant pour la comparaison avec la génération du rectangle correspondant aux buts qui vous est proposée.
    - Lancez le fichier ./src/compute_goal_surface.py dans un environnement Python.
        - Sur l'image, les buts détectés par notre algorithme apparaîtront en bleu, tandis que les buts que vous avez détectés seront en rouge.
        - Dans le terminal, vous obtiendrez diverses métriques permettant d'évaluer notre algorithme, telles que sa précision par rapport à votre prédiction manuelle, ou encore l'indice de Jaccard associé. L'indice de Jaccard correspond à l'aire de l'intersection des deux rectangles divisée par l'aire de l'union des deux rectangles. En notant A l'aire du rectangle issu de la prédiction algorithmique des buts et M l'aire du rectangle de la détection manuelle, on a :
        Jaccard(A,M) = |A ∩ M| / |A ∪ M|
