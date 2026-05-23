# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: geoffrey <geoffrey@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/05/15 12:07:08 by gematura          #+#    #+#              #
#    Updated: 2026/05/23 15:15:23 by geoffrey         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

BLUE = \033[1;34m
GREEN = \033[1;32m
RESET = \033[0m

ARGS ?=

.PHONY: install run debug clean fclean lint lint-strict test

install:
	@printf '$(BLUE)Installation des dépendances avec uv...$(RESET)\n'
	uv sync

run:
	@printf '$(BLUE)Lancement de Call me Maybe...$(RESET)\n'
	@printf '$(GREEN)Utilisation du modèle: Qwen/Qwen3-0.6B (par defaut)$(RESET)\n'
	uv run python -m src $(ARGS)

run-bloomz:
	@printf '$(BLUE)Lancement de Call me Maybe...$(RESET)\n'
	@printf '$(GREEN)Utilisation du modèle: bigscience/bloomz-560m$(RESET)\n'
	uv run python -m src --model "bigscience/bloomz-560m"$(ARGS)

run-smollm2:
	@printf '$(BLUE)Lancement de Call me Maybe...$(RESET)\n'
	@printf '$(GREEN)Utilisation du modèle: HuggingFaceTB/SmolLM2-360M-Instruct$(RESET)\n'
	uv run python -m src --model HuggingFaceTB/SmolLM2-360M-Instruct $(ARGS)

test:
	@printf '$(BLUE)Lancement de la suite de tests...$(RESET)\n'
	uv run python -m src --functions_definition data/test_input/functions_definition.json --input data/test_input/test_extrem_prompts.json --output data/output/test_extrem_results.json $(ARGS)
	uv run python -m src --functions_definition data/test_input/functions_stress_test.json --input data/test_input/prompts_stress_test.json --output data/output/test_stress_results.json $(ARGS)
	uv run python -m src --functions_definition data/test_input/test_complex_functions.json --input data/test_input/test_complex_prompts.json --output data/output/test_complex_results.json $(ARGS)
	-uv run python -m src --functions_definition data/test_input/functions_definition.json --input data/test_input/invalid_prompts_format.json $(ARGS)
	-uv run python -m src --functions_definition data/test_input/invalid_functions_format.json --input data/test_input/functions_calling_tests.json $(ARGS)
	@printf '$(GREEN)Tests terminés. Résultats disponibles dans data/output/$(RESET)\n'

debug:
	@printf '$(BLUE)Lancement en mode debug...$(RESET)\n'
	uv run python -m pdb -m src $(ARGS)

clean:
	@printf '$(BLUE)Nettoyage des fichiers temporaires...$(RESET)\n'
	rm -rf .mypy_cache .pytest_cache
	find . -type d -name '__pycache__' -exec rm -rf {} +
	uv clean --force

clean-test:
	@printf '$(BLUE)Nettoyage des fichiers tests$(RESET)\n'
	rm -rf data/output data/log
	

fclean: clean
	@printf "$(BLUE)Suppression de l'environnement virtuel...$(RESET)\n"
	rm -rf .venv data/output data/log
		
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

