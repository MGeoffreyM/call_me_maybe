"""Package principal de l'application Call Me Maybe.

Ce package implémente un moteur de *Function Calling* de niveau production,
conçu pour traduire des requêtes utilisateurs en langage naturel en appels de
fonctions JSON strictement typés et structurés, exploitables par des systèmes
tiers.

Il s'appuie sur une approche algorithmique de décodage sous contrainte
(Constrained Decoding) qui intercepte, filtre et applique un masque de
probabilités (-inf) sur les logits générés en temps réel par un modèle de
langage causal léger (Small LLM).
Piloté par une Machine à États Finis (FSM) et un système de pile de contextes,
le moteur garantit une sortie JSON syntaxiquement parfaite, immunisée contre
les hallucinations structurelles, et capable de gérer des structures d'objets
complexes ou imbriquées.

Caractéristiques clés :
    * Décodage guidé par FSM avec mise en cache optimisée des masques de
      vocabulaire.
    * Inférence parallélisée par multi-threading pour optimiser le traitement
      séquentiel imposé par les contraintes matérielles et logicielles
      (Batch Size 1).
    * Validation de type stricte et parsing sécurisé des fichiers de
      configuration et des entrées via Pydantic.
    * Télémétrie et journalisation avancée avec coloration ANSI et séparation
      des flux système et production.

Modules inclus :
    __main__ : Point d'entrée CLI, orchestration du pipeline de génération de
               bout en bout et distribution multithreadée des requêtes.
    constrainer : Cœur logique et algorithmique contenant les définitions des
                  états de la FSM (`JsonState`) et le contrôleur de logits
                  (`JsonConstraint`).
    parser : Module d'ingestion et de validation stricte des structures de
             données entrantes (définitions des fonctions et prompts de test).
    log_config : Configuration centralisée de la télémétrie, gestionnaires de
                 fichiers tournants (rotating) et formatage personnalisé
                 (niveau TOKEN).
"""
