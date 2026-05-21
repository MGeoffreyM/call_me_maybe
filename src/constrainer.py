"""Module de décodage sous contrainte (Constrained Decoding).

Implémente une Machine à États Finis (FSM) permettant d'imposer au modèle
la génération d'un JSON valide correspondant strictement aux fonctions
fournies.
"""

import sys
import json
import logging
from typing import Any
from enum import Enum, auto
import numpy as np
from pydantic import BaseModel, Field, model_validator
from .parser import Function


logger = logging.getLogger(__name__)


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
    quote_comma_tokens: list[int] = Field(default_factory=list)
    quote_brace_tokens: list[int] = Field(default_factory=list)

    current_state: JsonState = Field(default=JsonState.EXPECT_BRACE_OPEN)
    current_buffer: str = Field(default='')

    context_stack: list[dict[str, Any]] = Field(default_factory=list)

    def _real_str(self, text: str) -> str:
        """Convertit les symboles d'espacement des tokens en vrais espaces."""
        return text.replace('Ġ', ' ').replace('\u2581', ' ').replace('<0x20>',
                                                                     ' ')

    def push_context(self, function_name: str,
                     parameters: dict[str, Any]) -> None:
        """Sauvegarde l'état actuel et commence une nouvelle fonction."""
        self.context_stack.append({
            "name": function_name,
            "remaining": parameters.copy(),
            "current_key": "",
            "current_type": ""
        })

    def pop_context(self) -> None:
        """Termine la fonction actuelle et revient à la parente."""
        self.context_stack.pop()

    @property
    def current_context(self) -> dict[str, Any] | None:
        """Renvoie le contexte tout en haut de la pile."""
        return self.context_stack[-1] if self.context_stack else None

    @model_validator(mode='after')
    def extract_vocab_dictionary(self) -> 'JsonConstraint':
        """Extrait et classe les tokens utiles depuis le vocabulaire."""
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
                vocab = tokenizer_data["model"]["vocab"]
        except FileNotFoundError as e:
            logger.critical(f'Vocab Loading Error: {e}')
            sys.exit(1)

        func_names: list[str] = []
        all_key: list[str] = ['"name":"', '","parameters":{', ':']

        for func in self.allowed_functions:
            func_names.append(func.name)
            for key in func.parameters.keys():
                all_key.append(f'"{key}"')

        all_key = list(set(all_key))

        for sym in ['{', '}', '"', ':', ',']:
            if sym in vocab:
                self.json_symbols[sym] = vocab[sym]
            else:
                for t_str, t_id in vocab.items():
                    if self._real_str(t_str) == sym:
                        self.json_symbols[sym] = t_id
                        break

        for token_str, token_id in vocab.items():
            real_str = self._real_str(token_str)
            if real_str == "":
                continue

            if any(real_str in name for name in func_names):
                self.valid_name_tokens[token_str] = token_id

            if any(real_str in key for key in all_key):
                self.valid_key_token[token_str] = token_id

            if all(char in "0123456789.-" for char in real_str):
                self.valid_number_tokens[token_str] = token_id

            real_no_space = real_str.replace(' ', '')
            if '",' in real_no_space:
                self.quote_comma_tokens.append(token_id)
            if '"}' in real_no_space:
                self.quote_brace_tokens.append(token_id)

        return self

    def constrain_logits(self, logits: np.ndarray) -> np.ndarray:
        """Applique un masque d'infini négatif sur les logits interdits."""
        logits_contraints = np.full(logits.shape, -np.inf)

        match self.current_state:
            case JsonState.EXPECT_BRACE_OPEN:
                id_brace = self.json_symbols.get('{')
                if id_brace is not None:
                    logits_contraints[id_brace] = logits[id_brace]
                return logits_contraints

            case JsonState.EXPECT_NAME_KEY_PREFIX:
                target_string = '"name":"'
                if self.current_buffer == target_string:
                    return logits_contraints

                for token_str, token_id in self.valid_key_token.items():
                    real_token = self._real_str(token_str)
                    potential_buffer = self.current_buffer + real_token
                    if target_string.startswith(potential_buffer):
                        logits_contraints[token_id] = logits[token_id]
                return logits_contraints

            case JsonState.EXPECT_NAME_VALUE:
                if any(self.current_buffer == f.name for f
                       in self.allowed_functions):
                    id_quote = self.json_symbols.get('"')
                    if id_quote is not None:
                        logits_contraints[id_quote] = logits[id_quote]
                    return logits_contraints

                for token_str, token_id in self.valid_name_tokens.items():
                    real_token = self._real_str(token_str)
                    potential_buffer = self.current_buffer + real_token
                    if any(func.name.startswith(potential_buffer) for func
                           in self.allowed_functions):
                        logits_contraints[token_id] = logits[token_id]
                return logits_contraints

            case JsonState.EXPECT_PARAM_KEY_PREFIX:
                target_string = ',"parameters":{'
                if self.current_buffer == target_string:
                    return logits_contraints

                for token_str, token_id in self.valid_key_token.items():
                    real_token = self._real_str(token_str)
                    potential_buffer = self.current_buffer + real_token
                    if target_string.startswith(potential_buffer):
                        logits_contraints[token_id] = logits[token_id]
                return logits_contraints

            case JsonState.EXPECT_PARAM_KEY:
                ctx = self.current_context
                if ctx is None:
                    return logits_contraints
                if len(ctx["remaining"]) == 0:
                    id_brace = self.json_symbols.get('}')
                    if id_brace is not None:
                        logits_contraints[id_brace] = logits[id_brace]

                valid_targets = [f'"{k}"' for k in ctx["remaining"].keys()]

                for token_str, token_id in self.valid_key_token.items():
                    real_token = self._real_str(token_str)
                    potential_buffer = self.current_buffer + real_token
                    if any(target.startswith(potential_buffer) for target
                           in valid_targets):
                        logits_contraints[token_id] = logits[token_id]
                return logits_contraints

            case JsonState.EXPECT_PARAM_COLON:
                target_string = ':'
                if self.current_buffer == target_string:
                    return logits_contraints

                for token_str, token_id in self.valid_key_token.items():
                    real_token = self._real_str(token_str)
                    potential_buffer = self.current_buffer + real_token
                    if target_string.startswith(potential_buffer):
                        logits_contraints[token_id] = logits[token_id]
                return logits_contraints

            case JsonState.EXPECT_PARAM_VALUE:
                ctx = self.current_context
                if ctx is None:
                    return logits_contraints
                param_type = ctx["current_type"]
                remaining_params = ctx["remaining"]

                if param_type == 'object':
                    id_brace = self.json_symbols.get('{')
                    if id_brace is not None:
                        logits_contraints[id_brace] = logits[id_brace]

                elif param_type == 'number':
                    if len(self.current_buffer) < 40:
                        for token_str, token_id in \
                                self.valid_number_tokens.items():
                            real_token = self._real_str(token_str)
                            if '.' in self.current_buffer and '.' \
                                    in real_token:
                                continue
                            if '-' in real_token and len(
                                    self.current_buffer) > 0:
                                continue
                            logits_contraints[token_id] = logits[token_id]

                    has_digit = any(char.isdigit() for char
                                    in self.current_buffer)
                    if has_digit:
                        id_virgule = self.json_symbols.get(',')
                        id_brace = self.json_symbols.get('}')
                        if len(remaining_params) > 1 and id_virgule \
                                is not None:
                            logits_contraints[id_virgule] = logits[id_virgule]
                        elif len(remaining_params) == 1 and id_brace \
                                is not None:
                            logits_contraints[id_brace] = logits[id_brace]

                elif param_type == 'string':
                    if self.current_buffer == "":
                        id_quote = self.json_symbols.get('"')
                        if id_quote is not None:
                            logits_contraints[id_quote] = logits[id_quote]
                    else:
                        logits_contraints = np.copy(logits)

                        for t_id in self.quote_brace_tokens:
                            logits_contraints[t_id] = -np.inf
                        for t_id in self.quote_comma_tokens:
                            logits_contraints[t_id] = -np.inf

                elif param_type == 'boolean':
                    buffer_lower = self.current_buffer.lower().strip()
                    if buffer_lower in ['true', 'false']:
                        id_virgule = self.json_symbols.get(',')
                        id_brace = self.json_symbols.get('}')
                        if len(remaining_params) > 1 and id_virgule \
                                is not None:
                            logits_contraints[id_virgule] = logits[id_virgule]
                        elif len(remaining_params) == 1 and id_brace \
                                is not None:
                            logits_contraints[id_brace] = logits[id_brace]
                    else:
                        logits_contraints = np.copy(logits)
                else:
                    logits_contraints = np.copy(logits)

                return logits_contraints

            case JsonState.EXPECT_PARAM_NEXT:
                ctx = self.current_context
                if ctx is None:
                    return logits_contraints
                id_virgule = self.json_symbols.get(',')
                id_brace = self.json_symbols.get('}')

                if len(ctx["remaining"]) > 0 and id_virgule is not None:
                    logits_contraints[id_virgule] = logits[id_virgule]
                elif len(ctx["remaining"]) == 0 and id_brace is not None:
                    logits_contraints[id_brace] = logits[id_brace]
                return logits_contraints

            case JsonState.EXPECT_END_BRACE:
                id_brace = self.json_symbols.get('}')
                if id_brace is not None:
                    logits_contraints[id_brace] = logits[id_brace]
                return logits_contraints

            case JsonState.DONE:
                pass

        return logits

    def consume_token(self, token_id: int, token_str: str) -> None:
        """Met à jour l'état de la machine après qu'un token a été généré."""
        match self.current_state:
            case JsonState.EXPECT_BRACE_OPEN:
                if '{' in token_str:
                    self.current_state = JsonState.EXPECT_NAME_KEY_PREFIX

            case JsonState.EXPECT_NAME_KEY_PREFIX:
                self.current_buffer += token_str
                if self.current_buffer == '"name":"':
                    self.current_state = JsonState.EXPECT_NAME_VALUE
                    self.current_buffer = ""

            case JsonState.EXPECT_NAME_VALUE:
                if token_str == '"':
                    self.current_state = JsonState.EXPECT_PARAM_KEY_PREFIX
                    for func in self.allowed_functions:
                        if func.name == self.current_buffer:
                            # CRÉATION DU NOUVEAU CONTEXTE DANS LA PILE
                            self.push_context(func.name, func.parameters)
                            break
                    self.current_buffer = ''
                else:
                    self.current_buffer += token_str

            case JsonState.EXPECT_PARAM_KEY_PREFIX:
                self.current_buffer += token_str
                if self.current_buffer == ',"parameters":{':
                    self.current_state = JsonState.EXPECT_PARAM_KEY
                    self.current_buffer = ""

            case JsonState.EXPECT_PARAM_KEY:
                ctx = self.current_context
                if ctx is None:
                    return
                if len(ctx["remaining"]) == 0 and '}' in token_str:
                    self.current_state = JsonState.EXPECT_END_BRACE
                    self.current_buffer = ""
                else:
                    self.current_buffer += token_str
                    valid_targets = [f'"{k}"' for k in ctx["remaining"].keys()]
                    if self.current_buffer in valid_targets:
                        self.current_state = JsonState.EXPECT_PARAM_COLON
                        ctx["current_key"] = self.current_buffer.strip('"')
                        ctx["current_type"] = ctx["remaining"][
                            ctx["current_key"]].type
                        self.current_buffer = ""

            case JsonState.EXPECT_PARAM_COLON:
                self.current_buffer += token_str
                if self.current_buffer == ':':
                    self.current_state = JsonState.EXPECT_PARAM_VALUE
                    self.current_buffer = ""

            case JsonState.EXPECT_PARAM_VALUE:
                ctx = self.current_context
                if ctx is None:
                    return

                param_type = ctx["current_type"]

                # 1. OBJET
                if param_type == 'object':
                    if '{' in token_str:
                        self.current_state = JsonState.EXPECT_NAME_KEY_PREFIX
                        self.current_buffer = ""

                elif param_type == 'string':
                    self.current_buffer += token_str
                    buffer_content = self.current_buffer[1:]
                    is_closed = buffer_content.endswith(
                        '"') and not buffer_content.endswith('\\"')

                    clean_token = token_str.replace(' ', '').replace(
                        '\n', '').replace('\r', '')
                    if '",' in clean_token or '"}' in clean_token:
                        is_closed = True

                    if is_closed:
                        if ctx["current_key"] in ctx["remaining"]:
                            del ctx["remaining"][ctx["current_key"]]

                        if clean_token.endswith('","'):
                            self.current_state = JsonState.EXPECT_PARAM_KEY
                            self.current_buffer = '"'
                        elif clean_token.endswith('",'):
                            self.current_state = JsonState.EXPECT_PARAM_KEY
                            self.current_buffer = ""
                        elif clean_token.endswith('"}'):
                            self.current_state = JsonState.EXPECT_END_BRACE
                            self.current_buffer = ""
                        else:
                            self.current_state = JsonState.EXPECT_PARAM_NEXT
                            self.current_buffer = ""

                elif param_type in ['number', 'boolean']:
                    self.current_buffer += token_str
                    clean_token = token_str.replace(' ', '').replace(
                        '\n', '').replace('\r', '')

                    if ',' in clean_token or '}' in clean_token:
                        if ctx["current_key"] in ctx["remaining"]:
                            del ctx["remaining"][ctx["current_key"]]

                        if clean_token.endswith(',"'):
                            self.current_state = JsonState.EXPECT_PARAM_KEY
                            self.current_buffer = '"'
                        elif clean_token.endswith(','):
                            self.current_state = JsonState.EXPECT_PARAM_KEY
                            self.current_buffer = ""
                        elif clean_token.endswith('}'):
                            self.current_state = JsonState.EXPECT_END_BRACE
                            self.current_buffer = ""
                        else:
                            if '}' in clean_token:
                                self.current_state = JsonState.EXPECT_END_BRACE
                            else:
                                self.current_state = JsonState.EXPECT_PARAM_KEY
                            self.current_buffer = ""

            case JsonState.EXPECT_PARAM_NEXT:
                if ',' in token_str:
                    self.current_state = JsonState.EXPECT_PARAM_KEY
                elif '}' in token_str:
                    self.current_state = JsonState.EXPECT_END_BRACE

            case JsonState.EXPECT_END_BRACE:
                if '}' in token_str:
                    self.pop_context()

                    if self.current_context is not None:
                        parent_ctx = self.current_context
                        if parent_ctx["current_key"] in parent_ctx[
                                "remaining"]:
                            del parent_ctx["remaining"][parent_ctx[
                                "current_key"]]
                        self.current_state = JsonState.EXPECT_PARAM_NEXT
                    else:
                        self.current_state = JsonState.DONE

            case JsonState.DONE:
                pass

    def reset(self) -> None:
        """Réinitialise la machine à états pour un nouveau prompt."""
        self.current_state = JsonState.EXPECT_BRACE_OPEN
        self.current_buffer = ''
        self.context_stack = []
