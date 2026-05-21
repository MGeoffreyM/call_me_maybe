"""Module de parsing pour Call Me Maybe.

Gère la lecture et la validation stricte des fichiers JSON d'entrée
(définitions de fonctions et prompts de test) en utilisant Pydantic.
"""

from __future__ import annotations
import json
import sys
import logging
from pydantic import BaseModel, TypeAdapter, Field


logger = logging.getLogger(__name__)


class ParameterProperty(BaseModel):
    """Propriétés d'un paramètre de fonction (ex: type)."""
    type: str


class Function(BaseModel):
    """Modèle représentant la définition d'une fonction appelable."""
    name: str = ''
    description: str = ''
    parameters: dict[str, ParameterProperty]
    returns: ParameterProperty


class Prompt(BaseModel):
    """Modèle représentant une requête utilisateur de test."""
    prompt: str


class Parser(BaseModel):
    """Classe principale de chargement des données d'entrée."""
    list_function: list[Function] = Field(default_factory=list)
    list_prompt: list[Prompt] = Field(default_factory=list)

    def read_files(self, func_file: str, input_file: str) -> None:
        """Lit et valide les fichiers d'entrée JSON.

        Args:
            func_file: Le chemin vers le fichier de définition des fonctions.
            input_file: Le chemin vers le fichier des requêtes utilisateurs.

        """
        try:
            with open(func_file, 'r', encoding='utf-8') as f:
                data_func = json.load(f)
                self.list_function = TypeAdapter(
                    list[Function]).validate_python(data_func)

            with open(input_file, 'r', encoding='utf-8') as f:
                data_prompt = json.load(f)
                self.list_prompt = TypeAdapter(
                    list[Prompt]).validate_python(data_prompt)

        except Exception as e:
            logger.critical(f'Parsing Error: {e}')
            sys.exit(1)
