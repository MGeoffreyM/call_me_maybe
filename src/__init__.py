"""Package principal du projet Call Me Maybe.

Ce package implémente un système de *Function Calling* permettant de traduire
des requêtes en langage naturel en appels de fonctions JSON strictement
structurés.

Il s'appuie sur une approche de décodage sous contrainte (Constrained Decoding)
qui intercepte et masque les logits générés par un modèle de langage (LLM) en
temps réel. Grâce à une Machine à États Finis (FSM), le système garantit une
sortie JSON syntaxiquement parfaite et respectueuse des paramètres attendus.

Modules exportés :
    __main__ : Point d'entrée de l'application (CLI), gestion de la boucle de
               génération de bout en bout et interaction avec le SDK du LLM.
    constrainer : Cœur algorithmique contenant la FSM (`JsonState`) et la
                  logique de restriction des tokens (`JsonConstraint`).
    parser : Module de validation et de chargement des données (définitions
             des fonctions et requêtes de test) s'appuyant sur Pydantic.
"""
