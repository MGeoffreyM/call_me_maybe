*Ce projet a été créé dans le cadre du cursus 42 par geoffrey.*

# Call Me Maybe - LLM d'Appel de Fonction par Décodage Contraint

🌐 [English version](README.md)

## Description
**Call Me Maybe** est un orchestrateur d'appels de fonctions de haute fiabilité conçu autour de grands modèles de langage (LLM) compacts (par défaut : `Qwen/Qwen3-0.6B`). Dans les systèmes d'intelligence artificielle modernes, les modèles de langage (LLM) excellent dans le traitement des requêtes en langage naturel, mais sont intrinsèquement sujets aux hallucinations syntaxiques lorsqu'il leur est demandé de produire des structures informatiques strictes. Alors que les modèles commerciaux de pointe s'appuient sur un volume massif de paramètres pour maintenir le formatage, les modèles plus petits de moins d'un milliard de paramètres échouent généralement à générer du JSON structuré de manière cohérente.

Ce projet comble cette lacune en implémentant une couche rigide de **décodage contraint** (Constrained Decoding) au niveau des tokens, pilotée par une **machine à états finis (FSM)** déterministe. En remplaçant les probabilités mathématiques brutes (`logits`) avant qu'un token ne soit échantillonné, cette application force le moteur de langage à agir comme un analyseur (parser) structuré, garantissant **une sortie JSON syntaxiquement valide à 100 %** qui se conforme parfaitement à un schéma prédéfini en toutes circonstances.

---

## Instructions

### Prérequis
* **Système d'exploitation :** Linux ou macOS.
* **Environnement Python :** Python 3.10 ou supérieur.
* **Gestion des paquets :** `uv` (requis pour une synchronisation du projet ultra-rapide et isolée).

### Installation
La configuration du projet est entièrement automatisée via le `Makefile` fourni. Pour préparer l'environnement, créer l'environnement virtuel isolé et installer toutes les dépendances déclarées, exécutez :

```bash
make install
```

### Exécution & Règles Make
Le projet inclut un `Makefile` robuste pour une exécution simplifiée.

**Exécution standard (Qwen3-0.6B) :**
```bash
make run
```

**Exécution avec des modèles alternatifs (Bonus) :**
```bash
make run-bloomz    # Exécute avec bigscience/bloomz-560m
make run-smollm2   # Exécute avec HuggingFaceTB/SmolLM2-360M-Instruct
```

**Exécuter la suite de tests complète (Bonus) :**
```bash
make test
```

**Qualité du code & Linting :**
```bash
make lint          # Lance les vérifications flake8 et mypy
make lint-strict   # Lance une validation mypy stricte
```

---

## Exemple d'utilisation

### Ajustements dynamiques du pipeline via le CLI
Le système expose une interface en ligne de commande (CLI) permettant aux développeurs de remapper dynamiquement les schémas, les manifestes d'entrée, les destinations de sortie et le matériel de calcul, en suivant strictement les exigences du sujet :

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json \
  --device cuda
```

### Alignement structurel entre l'entrée et la sortie
Étant donné un élément de test typique dans `function_calling_tests.json` :

```json
{
  "prompt": "What is the sum of 265 and 345?"
}
```

L'application interprète les capacités disponibles dans `functions_definition.json` et crée une entrée structurée dans le fichier de sortie :

```json
[
  {
    "prompt": "What is the sum of 265 and 345?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 265.0,
      "b": 345.0
    }
  }
]
```

---

## Fonctionnalités Bonus Implémentées

Ce projet implémente avec succès les fonctionnalités bonus suivantes :

1. **Support de plusieurs modèles LLM :** Intégration de `MODEL_PROFILES` avec des modèles de prompt personnalisés pour supporter dynamiquement `bigscience/bloomz-560m` et `HuggingFaceTB/SmolLM2-360M-Instruct`.
2. **Suite de tests exhaustive :** Développement de prompts extrêmes, de tests de stress, de tests d'objets imbriqués complexes et de tests de validation de format (entièrement automatisés via `make test`).
3. **Visualisation du processus de génération :** Implémentation d'un `AppLogger` avec un codage couleur ANSI et un niveau de log personnalisé `TOKEN_LEVEL` pour visualiser l'état de la FSM, le buffer et le token exact généré en temps réel.
4. **Support des arguments de fonction imbriqués complexes :** Implémentation d'une pile de contexte (`self.push_context()` / `self.pop_context()`) dans la FSM pour supporter le suivi d'objets récursif et la génération de dictionnaires imbriqués.
5. **Mécanismes avancés de récupération d'erreurs :** Ajout d'un "Disjoncteur" (Circuit Breaker) de sécurité absolue pour détecter les boucles de génération de tokens répétitives et forcer la fermeture des tokens afin d'éviter les hallucinations infinies.

---

## Explication de l'algorithme

L'épine dorsale architecturale est une **machine à états finis (FSM)** déterministe définie dans `JsonState`. L'application fonctionne simultanément avec les couches neuronales du modèle, évaluant les configurations d'état token par token.

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

**La boucle de modification des Logits :**
1. Le programme appelle `get_logits_from_input_ids` pour récupérer les scores de probabilité bruts (`logits`) pour l'ensemble du vocabulaire.
2. Un tableau de masque d'isolement est préparé, pré-rempli avec l'infini négatif (`-np.inf`).
3. Sur la base de `self.current_state` et `self.current_buffer`, l'algorithme identifie les tokens légaux (par exemple, correspondance de préfixe pour `"true"` ou `"false"` en mode booléen).
4. Les logits d'origine des tokens valides sont copiés dans le masque.
5. `np.argmax()` sélectionne le meilleur token. Étant donné que toutes les transitions illégales restent à `-np.inf`, le modèle est mathématiquement incapable de violer la structure JSON.

---

## Décisions de Conception

### 1. Pré-filtrage granulaire du vocabulaire
Itérer sur le vocabulaire de 151 936 tokens pour chaque génération de token introduit une surcharge insoutenable. Pour résoudre ce problème, `JsonConstraint` exécute une phase de compilation hautement optimisée au sein d'un `@model_validator` Pydantic. Il remplit des tables de recherche dédiées et légères (`valid_name_tokens`, `valid_key_tokens`, `valid_boolean_tokens`), réduisant considérablement le temps de la boucle opérationnelle.

### 2. Pile de contexte pour les objets imbriqués
Pour gérer l'exigence bonus des objets imbriqués complexes, la FSM utilise une pile (`context_stack`). Lorsqu'un type `object` est rencontré, la FSM empile le contexte actuel et redémarre la logique d'analyse clé/valeur de manière récursive, en ne dépilant le contexte que lorsque le `}` imbriqué est généré avec succès.

### 3. Recyclabilité de la FSM
Une nouvelle instance du moteur de contraintes N'EST PAS générée pour chaque prompt. Au lieu de cela, l'orchestrateur charge le vocabulaire exactement une fois lors de l'initialisation et expose une méthode `.reset()` à grande vitesse qui vide les buffers et les piles de contexte entre les prompts.

---

## Analyse des Performances

* **Taux de conformité syntaxique :** Obtention d'un **score de validité d'analyse structurelle de 100 %**. Chaque fichier de sortie est analysé nativement via `json.loads()`.
* **Précision du routage :** Maintien d'un **taux de réussite du mappage des paramètres >90 %**.
* **Vitesse :** Grâce au pré-filtrage du vocabulaire, la vitesse d'inférence est strictement limitée par la passe avant (forward pass) du LLM, n'introduisant pratiquement aucune surcharge Python dans la boucle de génération. L'accélération matérielle via `--device cuda` réduit le temps d'inférence de plusieurs minutes (CPU) à quelques secondes.

---

## Défis Rencontrés

### 1. Anomalies d'agrégation des limites du tokenizer ("Méga-Tokens")
**Problème :** Le modèle produit occasionnellement des motifs de fin composites dans un seul bloc de token (par exemple, `'"}}\n'`). Les FSM standards se bloquent car elles s'attendent à ce que ces marqueurs de limite soient des tokens isolés.
**Solution :** Développement d'un système d'analyse par division. Lorsqu'une limite de fermeture est détectée à l'intérieur d'un token entrant, le reste est évalué dynamiquement, faisant passer l'état directement à `EXPECT_END_BRACE` ou `DONE`.

### 2. Le piège de l'hallucination booléenne
**Problème :** En attendant un booléen, si la FSM vérifie les mots exacts `"true"` ou `"false"` pour valider l'entrée, le LLM est libre de générer n'importe quelle chaîne aléatoire entre-temps, ce qui casse le JSON.
**Solution :** Implémentation d'une stricte correspondance de préfixes. Les logits sont mathématiquement restreints *uniquement* aux tokens qui commencent par les lettres de "true" ou "false" (par exemple, 't', 'tr', 'f', 'fa').

### 3. Violations JSON dues aux virgules de fin (Trailing Comma)
**Problème :** Le modèle tentait de générer des virgules de fin (`{"a": 16,}`).
**Solution :** Connexion de la logique d'état `EXPECT_PARAM_VALUE` directement dans le traqueur de variables actives. S'il ne reste qu'un seul paramètre attendu, le token virgule est explicitement supprimé, et `}` est verrouillé comme la seule option légale.

---

## Stratégie de Test

L'exactitude du système a été validée via un pipeline multicouche :
1. **Validation Pydantic :** Tous les fichiers d'entrée sont strictement validés avant traitement.
2. **Stress & Prompts Extrêmes :** Testé avec des nombres immenses, des flottants négatifs, des caractères spéciaux et des configurations d'expressions régulières (`data/test_input/test_extrem_prompts.json`).
3. **Gestion des formats invalides :** Confirmation que le système gère de manière élégante les schémas JSON corrompus (`invalid_functions_format.json`) et les clés manquantes via des blocs `try-except` robustes, sans jamais planter de manière inattendue.

---

## Ressources & Déclaration d'utilisation de l'IA

### Documents de Référence
* **SDK Hugging Face Transformers :** Conseils sur les matrices de traitement des logits.
* **Standards PEP 257 & PEP 484 :** Suivis pour des configurations de docstrings propres et un typage strict.

### Déclaration de Collaboration avec l'IA
L'IA a été utilisée comme auditeur de code interactif et conseiller en ingénierie système. Les points d'intégration spécifiques incluaient :
* L'affinage de la logique stricte de correspondance de préfixes booléens de la machine à états pour garantir une sortie de schéma 100 % valide.
* L'audit des opérations de masque de logit pour identifier les cas limites (edge cases) avec des formats de nombres négatifs et des virgules de fin.
* La structuration de ce README pour documenter correctement les comportements complexes de la FSM et les fonctionnalités bonus, conformément aux pratiques standard de documentation pour les développeurs.