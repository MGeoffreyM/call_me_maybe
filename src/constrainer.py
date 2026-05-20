"""Module de décodage sous contrainte (Constrained Decoding).

Implémente une Machine à États Finis (FSM) permettant d'imposer au modèle
la génération d'un JSON valide correspondant strictement aux fonctions
fournies.
"""

import sys
import json
from enum import Enum, auto
from typing import Any
import numpy as np
from pydantic import BaseModel, Field, model_validator
from .parser import Function


class JsonState(Enum):
    """États possibles de la FSM lors de la génération du JSON."""
    EXPECT_BRACE_OPEN = auto()
    EXPECT_NAME_KEY_PREFIX = auto()
    EXPECT_NAME_VALUE = auto()
    EXPECT_PARAM_KEY_PREFIX = auto()
    EXPECT_PARAM_KEY = auto()
    EXPECT_PARAM_COLON = auto()
    EXPECT_PARAM_VALUE = auto()
    EXPECT_PARAM_NEXT = auto()
    EXPECT_END_BRACE = auto()
    DONE = auto()


class JsonConstraint(BaseModel):
    """Contrôleur de logits basé sur la grammaire des fonctions autorisées."""
    vocab_path: str
    allowed_functions: list[Function]

    valid_name_tokens: dict[str, int] = Field(default_factory=dict)
    json_symbols: dict[str, int] = Field(default_factory=dict)
    valid_key_token: dict[str, int] = Field(default_factory=dict)
    valid_number_tokens: dict[str, int] = Field(default_factory=dict)

    current_state: JsonState = Field(default=JsonState.EXPECT_BRACE_OPEN)
    current_buffer: str = Field(default='')

    remaining_params: dict[str, Any] = Field(default_factory=dict)
    current_param_name: str = ''
    current_param_type: str = ''

    @model_validator(mode='after')
    def extract_vocab_dictionary(self) -> 'JsonConstraint':
        """Extrait et classe les tokens utiles depuis le vocabulaire."""
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)

        except FileNotFoundError as e:
            print(f'\033[1;31mVocab Loading Error: {e}\033[0m',
                  file=sys.stderr)
            sys.exit(1)

        func_names: list[str] = []
        all_key: list[str] = ['"name": "', '", "parameters": {', ': ']

        for func in self.allowed_functions:
            func_names.append(func.name)
            for key in func.parameters.keys():
                all_key.append(f'"{key}"')

        all_key = list(set(all_key))

        for token_str, token_id in vocab.items():
            clean_str_for_check = token_str.replace('Ġ', ' ')
            if any(clean_str_for_check in name for name in func_names):
                self.valid_name_tokens[token_str] = token_id

            if any(clean_str_for_check in key for key in all_key):
                self.valid_key_token[token_str] = token_id

            clean_token = token_str.replace('Ġ', '').replace(' ', '')
            if all(char in "0123456789.-+*/" for char in clean_token) \
                    and clean_token != "":
                self.valid_number_tokens[token_str] = token_id

            for sym in ['{', '}', '"', ':', ',', ' ', '{"']:
                if sym in vocab:
                    self.json_symbols[sym] = vocab[sym]

        return self

    def constrain_logits(self, logits: np.ndarray) -> np.ndarray:
        """Applique un masque d'infini négatif sur les logits interdits.

        Args:
            logits (np.ndarray): Le tableau des logits bruts en sortie du LLM.

        Returns:
            np.ndarray: Le tableau des logits modifiés selon la contrainte FSM.
        """
        logits_contraints = np.full(logits.shape, -np.inf)

        match self.current_state:
            case JsonState.EXPECT_BRACE_OPEN:
                id_brace = self.json_symbols.get('{')
                if id_brace:
                    logits_contraints[id_brace] = logits[id_brace]
                return logits_contraints

            case JsonState.EXPECT_NAME_KEY_PREFIX:
                target_string = '"name": "'
                if self.current_buffer == target_string:
                    return logits_contraints

                for token_str, token_id in self.valid_key_token.items():
                    clean_str = token_str.replace('Ġ', ' ')
                    potential_buffer = self.current_buffer + clean_str
                    if target_string.startswith(potential_buffer):
                        logits_contraints[token_id] = logits[token_id]
                return logits_contraints

            case JsonState.EXPECT_NAME_VALUE:
                if any(self.current_buffer == f.name for f in
                       self.allowed_functions):
                    id_quote = self.json_symbols.get('"')
                    if id_quote is not None:
                        logits_contraints[id_quote] = logits[id_quote]
                    return logits_contraints

                for token_str, token_id in self.valid_name_tokens.items():
                    clean_str = token_str.replace('Ġ', ' ')
                    potential_buffer = self.current_buffer + clean_str
                    if any(func.name.startswith(potential_buffer) for func in
                           self.allowed_functions):
                        logits_contraints[token_id] = logits[token_id]
                return logits_contraints

            case JsonState.EXPECT_PARAM_KEY_PREFIX:
                target_string = ', "parameters": {'
                if self.current_buffer == target_string:
                    return logits_contraints

                for token_str, token_id in self.valid_key_token.items():
                    clean_str = token_str.replace('Ġ', ' ')
                    potential_buffer = self.current_buffer + clean_str
                    if target_string.startswith(potential_buffer):
                        logits_contraints[token_id] = logits[token_id]
                return logits_contraints

            case JsonState.EXPECT_PARAM_KEY:
                if len(self.remaining_params) == 0:
                    id_brace = self.json_symbols.get('}')
                    if id_brace:
                        logits_contraints[id_brace] = logits[id_brace]
                valid_targets = [
                    f'"{k}"' for k in self.remaining_params.keys()]

                for token_str, token_id in self.valid_key_token.items():
                    clean_str = token_str.replace('Ġ', ' ')
                    potential_buffer = self.current_buffer + clean_str

                    if any(
                            target.startswith(potential_buffer)
                            for target in valid_targets):
                        logits_contraints[token_id] = logits[token_id]
                return logits_contraints

            case JsonState.EXPECT_PARAM_COLON:
                target_string = ': '
                if self.current_buffer == target_string:
                    return logits_contraints

                for token_str, token_id in self.valid_key_token.items():
                    clean_str = token_str.replace('Ġ', ' ')
                    potential_buffer = self.current_buffer + clean_str
                    if target_string.startswith(potential_buffer):
                        logits_contraints[token_id] = logits[token_id]
                return logits_contraints

            case JsonState.EXPECT_PARAM_VALUE:
                if self.current_param_type == 'number':
                    for token_id in self.valid_number_tokens.values():
                        logits_contraints[token_id] = logits[token_id]
                    has_digit = any(char.isdigit() for char
                                    in self.current_buffer)
                    if has_digit:
                        id_virgule = self.json_symbols.get(',')
                        id_brace = self.json_symbols.get('}')
                        if len(self.remaining_params) > 1 and id_virgule:
                            logits_contraints[id_virgule] = logits[id_virgule]
                        elif len(self.remaining_params) == 1 and id_brace:
                            logits_contraints[id_brace] = logits[id_brace]

                elif self.current_param_type == 'string':
                    if self.current_buffer == "":
                        id_quote = self.json_symbols.get('"')
                        if id_quote:
                            logits_contraints[id_quote] = logits[id_quote]
                    else:
                        logits_contraints = np.copy(logits)
                else:
                    logits_contraints = np.copy(logits)
                return logits_contraints

            case JsonState.EXPECT_PARAM_NEXT:
                id_virgule = self.json_symbols.get(',')
                id_brace = self.json_symbols.get('}')

                if len(self.remaining_params) > 0 and id_virgule:
                    logits_contraints[id_virgule] = logits[id_virgule]
                elif len(self.remaining_params) == 0 and id_brace:
                    logits_contraints[id_brace] = logits[id_brace]

                return logits_contraints

            case JsonState.EXPECT_END_BRACE:
                id_brace = self.json_symbols.get('}')
                if id_brace:
                    logits_contraints[id_brace] = logits[id_brace]
                return logits_contraints

            case JsonState.DONE:
                pass

        return logits

    def consume_token(self, token_id: int, token_str: str) -> None:
        """Met à jour l'état de la machine après qu'un token a été généré.

        Args:
            token_id (int): L'identifiant du token dans le vocabulaire.
            token_str (str): La représentation textuelle du token.
        """
        match self.current_state:

            case JsonState.EXPECT_BRACE_OPEN:
                if '{' in token_str:
                    self.current_state = JsonState.EXPECT_NAME_KEY_PREFIX

            case JsonState.EXPECT_NAME_KEY_PREFIX:
                self.current_buffer += token_str

                if self.current_buffer == '"name": "':
                    self.current_state = JsonState.EXPECT_NAME_VALUE
                    self.current_buffer = ""

            case JsonState.EXPECT_NAME_VALUE:
                if token_str == '"':
                    self.current_state = JsonState.EXPECT_PARAM_KEY_PREFIX
                    for func in self.allowed_functions:
                        if func.name == self.current_buffer:
                            self.remaining_params = func.parameters.copy()
                            break
                    self.current_buffer = ''
                else:
                    self.current_buffer += token_str

            case JsonState.EXPECT_PARAM_KEY_PREFIX:
                self.current_buffer += token_str

                if self.current_buffer == ', "parameters": {':
                    self.current_state = JsonState.EXPECT_PARAM_KEY
                    self.current_buffer = ""

            case JsonState.EXPECT_PARAM_KEY:
                if len(self.remaining_params) == 0 and '}' in token_str:
                    self.current_state = JsonState.EXPECT_END_BRACE
                    self.current_buffer = ""
                else:
                    self.current_buffer += token_str

                valid_targets = [
                    f'"{k}"' for k in self.remaining_params.keys()]

                if self.current_buffer in valid_targets:
                    self.current_state = JsonState.EXPECT_PARAM_COLON
                    self.current_param_name = self.current_buffer.strip('"')
                    self.current_param_type = self.remaining_params[
                        self.current_param_name].type
                    self.current_buffer = ""

            case JsonState.EXPECT_PARAM_COLON:
                self.current_buffer += token_str

                if self.current_buffer == ': ':
                    self.current_state = JsonState.EXPECT_PARAM_VALUE
                    self.current_buffer = ""

            case JsonState.EXPECT_PARAM_VALUE:
                if self.current_param_type == 'number':
                    if token_str in [',', '}']:
                        if self.current_param_name in self.remaining_params:
                            del self.remaining_params[self.current_param_name]
                        self.current_buffer = ""

                        if token_str == "}":
                            self.current_state = JsonState.EXPECT_END_BRACE
                        else:
                            self.current_state = JsonState.EXPECT_PARAM_KEY
                    else:
                        self.current_buffer += token_str

                elif self.current_param_type == "string":
                    if '"' in token_str and self.current_buffer != "":
                        if self.current_param_name in self.remaining_params:
                            del self.remaining_params[self.current_param_name]

                        self.current_buffer = ""
                        after_quote = token_str.split('"', 1)[1]
                        if after_quote.count('}') >= 2:
                            self.current_state = JsonState.DONE
                        elif after_quote.count('}') == 1:
                            self.current_state = JsonState.EXPECT_END_BRACE
                        elif ',' in after_quote:
                            self.current_state = JsonState.EXPECT_PARAM_KEY
                        else:
                            self.current_state = JsonState.EXPECT_PARAM_NEXT
                    else:
                        self.current_buffer += token_str
                else:
                    if ',' in token_str or '}' in token_str:
                        if self.current_param_name in self.remaining_params:
                            del self.remaining_params[self.current_param_name]
                        self.current_buffer = ""

                        if '}' in token_str:
                            self.current_state = JsonState.EXPECT_END_BRACE
                        else:
                            self.current_state = JsonState.EXPECT_PARAM_KEY
                    else:
                        self.current_buffer += token_str

            case JsonState.EXPECT_PARAM_NEXT:
                if token_str == ',':
                    self.current_state = JsonState.EXPECT_PARAM_KEY
                elif token_str == '}':
                    self.current_state = JsonState.EXPECT_END_BRACE

            case JsonState.EXPECT_END_BRACE:
                if token_str == '}':
                    self.current_state = JsonState.DONE

            case JsonState.DONE:
                pass

    def reset(self) -> None:
        """Réinitialise la machine à états pour un nouveau prompt."""
        self.current_state = JsonState.EXPECT_BRACE_OPEN
        self.current_buffer = ''
        self.remaining_params = {}
        self.current_param_name = ''
        self.current_param_type = ''
