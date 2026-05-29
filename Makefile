# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: geoffrey <geoffrey@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/05/15 12:07:08 by gematura          #+#    #+#              #
#    Updated: 2026/05/29 12:31:00 by geoffrey         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

BLUE = \033[1;34m
GREEN = \033[1;32m
RESET = \033[0m

ARGS ?=

.PHONY: install debug clean fclean lint lint-strict run test \
		bloomz smollm2 custom extrem stress complex
		
# --- OPTIONS DES MODÈLES ET TOKENIZER (RUN & TEST COUPLÉS) ---
MODEL_FLAG =
MODEL_NAME = Qwen/Qwen3-0.6B (par défaut)
TOKEN_FLAG =

ifneq ($(filter bloomz,$(MAKECMDGOALS)),)
    MODEL_FLAG = --model "bigscience/bloomz-560m"
    MODEL_NAME = bigscience/bloomz-560m
endif

ifneq ($(filter smollm2,$(MAKECMDGOALS)),)
    MODEL_FLAG = --model "HuggingFaceTB/SmolLM2-360M-Instruct"
    MODEL_NAME = HuggingFaceTB/SmolLM2-360M-Instruct
endif

ifneq ($(filter custom,$(MAKECMDGOALS)),)
    TOKEN_FLAG = --custom-tokenizer
endif

# --- OPTIONS DU COMPORTEMENT DE LA RÈGLE TEST ---
RUN_EXTREM =
RUN_STRESS =
RUN_COMPLEX =
RUN_INVALID =
HAS_TEST_OPT =

ifneq ($(filter extrem,$(MAKECMDGOALS)),)
    RUN_EXTREM = 1
    HAS_TEST_OPT = 1
endif

ifneq ($(filter stress,$(MAKECMDGOALS)),)
    RUN_STRESS = 1
    HAS_TEST_OPT = 1
endif

ifneq ($(filter complex,$(MAKECMDGOALS)),)
    RUN_COMPLEX = 1
    HAS_TEST_OPT = 1
endif

# Si aucune option de filtrage spécifique n'est passée à "test", on exécute tout par défaut
ifeq ($(HAS_TEST_OPT),)
    RUN_EXTREM = 1
    RUN_STRESS = 1
    RUN_COMPLEX = 1
    RUN_INVALID = 1
endif

# Cibles fictives vides pour empêcher Make de lever une erreur de cible inconnue
bloomz smollm2 custom extrem stress complex:
	@:

install:
	@printf '$(BLUE)Installation des dépendances avec uv...$(RESET)\n'
	uv sync

# --- RÈGLE RUN UNIFIÉE ---
run:
	@printf '$(BLUE)Lancement de Call me Maybe...$(RESET)\n'
	@printf '$(GREEN)Modèle sélectionné : $(MODEL_NAME)$(RESET)\n'
	@if [ -n "$(TOKEN_FLAG)" ]; then \
		printf '$(GREEN)Tokenizer sélectionné : Custom Greedy Tokenizer$(RESET)\n'; \
	fi
	uv run python -m src $(MODEL_FLAG) $(TOKEN_FLAG) $(ARGS)

# --- RÈGLE TEST DYNAMIQUE ET CUMULABLE ---
test:
	@printf '$(BLUE)Lancement de la suite de tests...$(RESET)\n'
	@if [ -n "$(RUN_EXTREM)" ]; then \
		printf '$(BLUE)-> Exécution du test EXTREM$(RESET)\n'; \
		uv run python -m src --functions_definition data/test_input/functions_definition.json --input data/test_input/test_extrem_prompts.json --output data/output/test_extrem_results.json $(MODEL_FLAG) $(TOKEN_FLAG) $(ARGS); \
	fi
	@if [ -n "$(RUN_STRESS)" ]; then \
		printf '$(BLUE)-> Exécution du test STRESS$(RESET)\n'; \
		uv run python -m src --functions_definition data/test_input/test_stress_functions.json --input data/test_input/test_stress_prompts.json --output data/output/test_stress_results.json $(MODEL_FLAG) $(TOKEN_FLAG) $(ARGS); \
	fi
	@if [ -n "$(RUN_COMPLEX)" ]; then \
		printf '$(BLUE)-> Exécution du test COMPLEX$(RESET)\n'; \
		uv run python -m src --functions_definition data/test_input/test_complex_functions.json --input data/test_input/test_complex_prompts.json --output data/output/test_complex_results.json $(MODEL_FLAG) $(TOKEN_FLAG) $(ARGS); \
	fi
	@if [ -n "$(RUN_INVALID)" ]; then \
		printf '$(BLUE)-> Exécution des tests de robustesse (Formats Invalides)$(RESET)\n'; \
		-uv run python -m src --input data/test_input/invalid_prompts_format.json $(MODEL_FLAG) $(TOKEN_FLAG) $(ARGS); \
		-uv run python -m src --functions_definition data/test_input/invalid_functions_format.json $(MODEL_FLAG) $(TOKEN_FLAG) $(ARGS); \
	fi
	@printf '$(GREEN)Tests terminés. Résultats disponibles dans data/output/$(RESET)\n'

debug:
	@printf '$(BLUE)Lancement en mode debug...$(RESET)\n'
	uv run python -m pdb -m src $(ARGS)

clean:
	@printf '$(BLUE)Nettoyage des fichiers temporaires...$(RESET)\n'
	rm -rf .mypy_cache .pytest_cache
	find . -type d -name '__pycache__' -exec rm -rf {} +
	uv clean
	@printf '$(BLUE)Nettoyage des fichiers tests$(RESET)\n'
	rm -rf data/output data/log
	@printf '$(GREEN)Fichiers nettoyés!$(RESET)\n'

fclean: clean
	@printf "$(BLUE)Suppression de l'environnement virtuel...$(RESET)\n"
	rm -rf .venv
	@printf '$(GREEN)Fichiers nettoyés!$(RESET)\n'

lint:
	@printf '$(BLUE)Lancement de flake8...$(RESET)\n'
	uv run flake8 .
	@printf '$(BLUE)Lancement de mypy ...$(RESET)\n'
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs .
	@printf '$(GREEN)MYPY et Flake8 terminés. Aucune erreur trouvée.$(RESET)\n'

lint-strict:
	@printf '$(BLUE)Lancement de flake8...$(RESET)\n'
	uv run flake8 .
	@printf '$(BLUE)Lancement de mypy (mode strict)...$(RESET)\n'
	uv run mypy --strict .
	@printf '$(GREEN)MYPY et Flake8 terminés en mode STRICT. Aucune erreur trouvée.$(RESET)\n'
