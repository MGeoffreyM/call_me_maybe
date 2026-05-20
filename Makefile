# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: geoffrey <geoffrey@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/05/15 12:07:08 by gematura          #+#    #+#              #
#    Updated: 2026/05/20 22:28:17 by geoffrey         ###   ########.fr        #
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
	uv run python -m src $(ARGS)

debug:
	@printf '$(BLUE)Lancement en mode debug...$(RESET)\n'
	uv run python -m pdb -m src $(ARGS)

clean:
	@printf '$(BLUE)Nettoyage des fichiers temporaires...$(RESET)\n'
	rm -rf .mypy_cache .pytest_cache
	find . -type d -name '__pycache__' -exec rm -rf {} +
	uv clean --force

fclean: clean
	@printf "$(BLUE)Suppression de l'environnement virtuel...$(RESET)\n"
	rm -rf .venv data/output call_me_maybe.log
		
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

test:
	@printf '$(BLUE)Lancement de la suite de tests unitaires avec pytest...$(RESET)\n'
	uv run pytest tests/ -v
	@printf '$(GREEN)Tests unitaires terminés. Aucune erreur trouvée.$(RESET)\n'
