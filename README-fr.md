*Ce projet a été créé dans le cadre du cursus 42 par gematura.*

# Call Me Maybe - LLM d'Appel de Fonction par Décodage Contraint

**🌐 [English Version](README.md)**

## Description
**Call Me Maybe** est un orchestrateur d'appels de fonctions de haute fiabilité conçu autour de petits modèles de langage (Small LLMs) compacts comme `Qwen/Qwen3-0.6B`.

Dans les systèmes d'intelligence artificielle modernes, les modèles de langage (LLM) excellent dans le traitement des requêtes en langage naturel, mais sont intrinsèquement sujets aux hallucinations syntaxiques lorsqu'il leur est demandé de produire des structures informatiques strictes. Alors que les modèles commerciaux de pointe s'appuient sur un volume massif de paramètres pour maintenir le formatage, les modèles plus petits de moins d'un milliard de paramètres échouent généralement à générer du JSON structuré de manière cohérente.

Ce projet comble cette lacune en implémentant une couche rigide de **décodage contraint** (Constrained Decoding) au niveau des tokens, pilotée par une **machine à états finis (FSM)** déterministe. En remplaçant les probabilités mathématiques brutes (`logits`) avant qu'un token ne soit échantillonné, cette application force le moteur de langage à agir comme un analyseur (parser) structuré, garantissant **une sortie JSON syntaxiquement valide** qui se conforme parfaitement à un schéma prédéfini en toutes circonstances.

---

## Instructions
### Prérequis
* **Système d'exploitation :** Linux ou macOS.
* **Environnement Python :** Python 3.13 ou supérieur.
* **Gestion des paquets :** `uv` (Un gestionnaire de paquets Python ultra-rapide écrit en Rust, utilisé pour le suivi transparent des dépendances et l'isolation de l'environnement virtuel).

### Installation
Créez l'environnement virtuel et installez les dépendances:
```bash
make install
```

### Utilisation
* **Exécution standard :**
  ```bash
  # Exécution par défaut : modèle Qwen/Qwen3-0.6B avec la tokenisation native du SDK
  make run
  ```
* **Sélectionner un modèle de langage spécifique :**
  ```bash
  make run bloomz      # Utilise le modèle bigscience/bloomz-560m
  make run smollm2     # Utilise le modèle HuggingFaceTB/SmolLM2-360M-Instruct
  ```
* **Utiliser le Tokenizer personalisé :**
  Exécute Qwen avec le tokenizer personnalisé (Greedy Longest Match)
  ```bash
  make run custom
  ```
* **Combiner les modificateurs de modèle et de tokenizer :**
  ```bash
  make run smollm2 custom
  ```
* **Passer des chemins ou des flags personnalisés :**
  ```bash
  # Exemple:
  make run ARGS="--input custom_prompts.json --output custom_results.json"

  # Flags disponibles:
    --functions_definition <path>
    --input <path>
    --output <path>
    --device <cpu> <cuda> <mps>
  ```
* **Exécuter la série de tests :** 

  Le moteur intègre un pipeline de test complet qui prend en charge l'exécution isolée ou combinée de suites de tests, couplée à n'importe quel modèle ou tokenizer cible.
  ```bash
  # Exécuter TOUTES les suites de tests de toutes les catégories (par défaut)
  make test

  # Exécuter des suites de tests ciblées
  make test extrem       # Exécute les cas limites, les grands nombres et les bornes extrêmes
  make test stress       # Exécute les tests de distribution des invites sous haute charge
  make test complex      # Exécute les structures de fonctions à objets imbriqués profonds

  # Tests croisés avec les modèles et le bonus du tokenizer personnalisé
  make test complex custom
  make test stress smollm2
  make test extrem bloomz custom
  ```
* **Débogage :**

  Lance le programme avec `pdb`
  ```bash
  make debug  
  ```

* **Qualité du code & Linting :**
  ```bash
  make lint          # Lance les vérifications flake8 et mypy
  make lint-strict   # Lance une validation flake8 et mypy stricte
  ```
* **Nettoyage :**
  ```bash
  make clean  # Supprime les fichiers temporaires et les résultats.
  make fclean  # Execute clean et supprime en plus l'environnement virtuel.
  ```

---

## Algorithmes, Décisions et Performances
### Algorithme
* **FSM (machine à états finis) :** L'épine dorsale architecturale est une **machine à états finis (FSM)** déterministe définie dans `JsonState`. L'application fonctionne simultanément avec les couches neuronales du modèle, évaluant les configurations d'état token par token.

```text
       [EXPECT_BRACE_OPEN]      --> Force '{'
                |
     [EXPECT_NAME_KEY_PREFIX]   --> Force '"name":"'
                |
       [EXPECT_NAME_VALUE]      --> Limite le vocabulaire aux noms de fonctions valides
                |
    [EXPECT_PARAM_KEY_PREFIX]   --> Force '","parameters":{'
                |
       [EXPECT_PARAM_KEY]       --> Restreint le vocabulaire aux noms de propriétés non utilisés
                |
      [EXPECT_PARAM_COLON]      --> Force '":'
                |
      [EXPECT_PARAM_VALUE]      --> Vérifie dynamiquement le type des nombres, chaînes et booléens
                |
      [EXPECT_PARAM_NEXT]       --> Décide entre ',' (plus de clés) ou '}' (fermer le bloc)
```

* **La boucle de modification des Logits :**
1. Le programme appelle `get_logits_from_input_ids` pour récupérer les scores de probabilité bruts (`logits`) pour l'ensemble du vocabulaire.
2. Un tableau de masque d'isolement est préparé, pré-rempli avec l'infini négatif (`-np.inf`).
3. Sur la base de `self.current_state` et `self.current_buffer`, l'algorithme identifie les tokens légaux (par exemple, correspondance de préfixe pour `"true"` ou `"false"` en mode booléen).
4. Les logits d'origine des tokens valides sont copiés dans le masque.
5. `np.argmax()` sélectionne le meilleur token. Étant donné que toutes les transitions illégales restent à `-np.inf`, le modèle est mathématiquement incapable de violer la structure JSON.

### Choix de conceptions:
* **Pile de Contexte pour la Structure Récursive** : Pour prendre en charge l'imbrication multi-profonde, la FSM utilise une liste de suivi active (`context_stack`). Lorsqu'un paramètre de type `object` est rencontré et que son schéma contient des sous-propriétés (`properties`), un nouveau dictionnaire de sous-contexte est poussé sur la pile. Les accolades de fermeture (`}`) dépilent dynamiquement le contexte actuel pour revenir de manière transparente aux portées parentes.
* **Traitement Multi-thread :** Accélération de l'inférence séquentielle imposée par le batch size 1, via un traitement parallèle multithreadé (`concurrent.futures`).
* **Logging Avancé :** Système de logs personnalisé avec codes couleurs ANSI, séparation des logs de production (`prod_call_me_maybe.log`) et système (`sys_call_me_maybe.log`), ainsi qu'un niveau de trace spécifique pour surveiller la génération token par token.
* **Optimisation de la Mémoire par Cache de Tokens** : Les masques de contraintes sur les logits sont des tableaux lourds à construire à la volée. Nous avons conçu une table de cache statique (`_mem_cache`) indexant les combinaisons de `(current_state, current_buffer, remaining_keys)`. Cela évite de réallouer des masques lors de contextes d'exécution simultanés, accélérant massivement la vitesse de décodage.

### Analyse des Performances: 
Notre architecture a été conçue pour équilibrer la stricte validité structurelle et le temps d'inférence, en contournant les limites inhérentes au traitement token-by-token.
* **Fiabilité et Précision (Accuracy) :** Sans contrainte, un modèle de 0.5B paramètres échoue à générer un JSON valide dans une grande majorité des cas (hallucinations de clés, oublis d'accolades, types de données erronés). Avec notre FSM dynamique, la validité syntaxique et schématique atteint 100%. Chaque clé générée correspond strictement au dictionnaire `functions_definition.json`.
* **Surcharge de Calcul (Computational Overhead) :** L'évaluation de l'arbre syntaxique (FSM) introduit théoriquement une pénalité à chaque token généré. Cependant, grâce au pré-filtrage du vocabulaire lors de l'instanciation (création de listes restreintes comme `valid_name_tokens` ou `valid_key_tokens`) et au système de `_mem_cache`, l'opération de masquage des logits s'exécute en $O(1)$ dans la boucle principale. La vitesse de génération reste donc presque exclusivement dictée par le temps de la passe avant (forward pass) du LLM.
* **Empreinte Mémoire (Trade-off RAM vs Temps CPU) :** Le masquage nécessite la création et l'addition de tableaux NumPy (`logits.shape`, soit environ ~150 000 flottants par token selon le vocabulaire du modèle). Plutôt que de recréer ces tableaux, le `_mem_cache` les stocke par état. Cela consomme légèrement plus de RAM au fil de l'exécution, mais réduit drastiquement la charge CPU, empêchant les ralentissements sur les longues requêtes.
* **Scalabilité, Multithreading et Gestion du GIL (Global Interpreter Lock) :** Le décodage contraint impose d'utiliser un Batch Size de 1 au niveau du LLM, car chaque prompt suit un chemin d'état FSM unique et imprévisible. Pour traiter massivement les fichiers d'entrée sans subir cette linéarité, le pipeline distribue les requêtes via un ThreadPoolExecutor configuré à max_workers=4. Ce chiffre représente le compromis matériel et logiciel parfait :

  * Protection du GPU : Il s'aligne sur la bande passante mémoire (VRAM) disponible, évitant la saturation du matériel ou les erreurs Out Of Memory (OOM) qu'une concurrence trop agressive (ex: 16 ou 32 workers) provoquerait.

  * Parallélisme réel et relâchement du GIL : Bien que Python possède un verrou global (GIL) limitant l'exécution du code pur à un seul cœur CPU, les bibliothèques basées sur du C++ sous-jacent (comme PyTorch lors de la passe avant du modèle ou NumPy lors des calculs de tenseurs) relâchent explicitement le GIL. Les calculs neuronaux s'exécutent ainsi en parallèle et à pleine vitesse sur le GPU ou les cœurs CPU disponibles.

  * Prévention de la GIL Contention (GIL Thrashing) : À la fin de chaque passe avant, chaque thread doit impérativement réacquérir le GIL pour exécuter la logique sémantique de notre FSM en Python pur (constrainer.py). Limiter l'application à 4 threads actifs empêche un phénomène d'embouteillage massif (où les threads passent plus de temps à se battre pour le verrou qu'à avancer), garantissant une fluidité maximale. De plus, cela laisse les cœurs restants de la machine de test libres pour la stabilité du système d'exploitation.

Grâce à cette parallélisation optimisée, l'intégralité de la suite de tests (Stress Tests inclus) est traitée très largement sous la barre des 5 minutes réglementaires.

### Défis Rencontrés
Tokenisation BPE vs Perte Sémantique du Greedy Match
* **Défi** : Pour éliminer les dépendances directes au SDK, nous avons développé un `CustomTokenizer` public utilisant un algorithme *Greedy Longest Match*. Bien que structurellement parfait, il a révélé un défi de taille : les voies neuronales des LLM sont liées aux frontières de tokens spécifiques générées par leur découpage natif **Byte-Pair Encoding (BPE)** lors de l'entraînement.
* **Impact** : Des segments de texte comme `"schrek"` ont été découpés en sous-mots différents de ceux attendus par le modèle, entraînant une dégradation sémantique (par exemple, le modèle hallucinant `"frodo"` au lieu d'extraire le mot demandé).
* **Résolution** : Nous avons mis en place une séparation architecturale sécurisée. L'application exécute par défaut le décodage BPE natif de haute fidélité, tandis que le tokenizer Greedy personnalisé reste entièrement isolable via le drapeau CLI `--custom-tokenizer` et la commande `make run custom` pour l'évaluation pédagogique du bonus.

Anomalies d'agrégation des limites du tokenizer ("Méga-Tokens")
* **Problème :** Le modèle produit occasionnellement des motifs de fin composites dans un seul bloc de token (par exemple, `'"}}\n'`). Les FSM standards se bloquent car elles s'attendent à ce que ces marqueurs de limite soient des tokens isolés.
* **Solution :** Développement d'un système d'analyse par division. Lorsqu'une limite de fermeture est détectée à l'intérieur d'un token entrant, le reste est évalué dynamiquement, faisant passer l'état directement à `EXPECT_END_BRACE` ou `DONE`.

Violations JSON dues aux virgules de fin (Trailing Comma)
* **Problème :** Le modèle tentait de générer des virgules de fin (`{"a": 16,}`).
* **Solution :** Connexion de la logique d'état `EXPECT_PARAM_VALUE` directement dans le traqueur de variables actives. S'il ne reste qu'un seul paramètre attendu, le token virgule est explicitement supprimé, et `}` est verrouillé comme la seule option légale.

### Fonctionnalités Bonus Implémentées

Ce projet implémente les fonctionnalités bonus suivantes :

1. **Support de plusieurs modèles LLM :** Intégration de `MODEL_PROFILES` avec des modèles de prompt personnalisés pour supporter dynamiquement `bigscience/bloomz-560m` et `HuggingFaceTB/SmolLM2-360M-Instruct`.

2. **Suite de tests exhaustive :** Développement de prompts extrêmes, de tests de stress, de tests d'objets imbriqués complexes et de tests de validation de format (entièrement automatisés via `make test`).

3. **Visualisation du processus de génération :** Implémentation d'un `AppLogger` avec un codage couleur ANSI et un niveau de log personnalisé `TOKEN_LEVEL` pour visualiser l'état de la FSM, le buffer et le token exact généré en temps réel.

4. **Support des arguments de fonction imbriqués complexes :** Implémentation d'une pile de contexte (`self.push_context()` / `self.pop_context()`) dans la FSM pour supporter le suivi d'objets récursif et la génération de dictionnaires imbriqués.

5. **Optimisation des performances (caching, batching) :**
    * Caching (Optimisation CPU) : Le calcul des masques de logits (tableaux NumPy de probabilités négatives) est lourd. Le système met en cache chaque masque généré selon l'état exact de la FSM. Si une situation structurelle se représente, le masque est récupéré en $O(1)$ depuis la mémoire partagée.
    * Batching/Parallélisation (Optimisation I/O) : Au lieu de traiter les prompts séquentiellement, le moteur distribue les tâches sur plusieurs threads simultanés (multithreading à 4 workers), réduisant drastiquement le temps d'exécution global de la suite de tests.

6. **Mécanismes avancés de récupération d’erreur :**
Les erreurs de génération (ex: JSONDecodeError) sont interceptées individuellement dans process_single_prompt() et isolées, permettant au reste du batch multithreadé de se terminer sans crash. Le diagnostic est assuré par un système de logs robuste séparant les flux (prod_call_me_maybe.log vs sys_call_me_maybe.log) avec rotation de fichiers (fail-safe). La création d'un niveau de log sur-mesure (TOKEN, niveau 21) avec coloration ANSI permet de tracer visuellement et en temps réel la source exacte des erreurs d'hallucination au sein de la machine à états.

7. **Recodage du tokenizer &**
8. **implémentation publique de l'encodage et du décodage du tokenizer :**
Le tokenizer officiel du SDK a été entièrement recodé. Il n'est pas qu'une simple surcouche : il charge lui-même le vocabulaire JSON et expose deux méthodes publiques indépendantes : encode(text) et decode(token_ids). L'encodage utilise un algorithme robuste de Greedy Longest Match (recherche de la plus grande sous-chaîne correspondante en triant les tokens par longueur décroissante).

9. **Démonstration de l'intégration du codage/décodage avec le décodage contraint :**
L'intégration est démontrée via le flag --custom-tokenizer. Lorsqu'il est activé, le pipeline contourne totalement le SDK du modèle pour la gestion du texte. Le prompt textuel est converti en tenseur d'entrée par CustomTokenizer.encode(), et chaque token ID généré par l'inférence (np.argmax(logits)) est reconverti en texte par CustomTokenizer.decode() avant d'être injecté dans la machine à états (constrainer.consume_token()).

## Ressources
### Documents de Référence

* **Hugging Face :** Plateforme de référence pour les modèles Open Source, les Tokenizers et l'écosystème Transformers. [https://huggingface.co/](https://huggingface.co/)
* Documentation officielle de PyTorch.
* Documentation de Pydantic pour la validation des données en Python, la modélisation récursive des classes et la reconstruction au moment de l'exécution (`model_rebuild`).
* Documentation sur le POSIX Threads & GIL pour la modélisation conceptuelle des performances des applications multithreadées dans les conditions standard de l'interpréteur.

### Utilisation de l'IA

Au cours de ce projet, l'Intelligence Artificielle a été utilisée de manière ciblée en tant qu'assistant technique :
* **Apprentissage et compréhension :** Assimilation des nouveaux concepts (comme l'espace latent, les logits, et le Constrained Decoding).
* **Traduction :** Traduction et synthèse de différents documents techniques ou documentations de librairies.
* **Guide et brainstorming :** Réflexion sur l'architecture de la Machine à États Finis (FSM) et sur la logique de parcours d'arbres complexes.
* **Débogage :** Aide à l'identification et à la résolution de bugs rencontrés en cours de développement (gestion du typage strict, comportements inattendus du multithreading avec le GIL de Python).
* **Readme :** Rédaction et structuration de ce README pour documenter correctement les comportements complexes de la FSM et les fonctionnalités bonus, 
