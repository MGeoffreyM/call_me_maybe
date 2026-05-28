"""Implémentation d'un Tokenizer public et indépendant du SDK (Pydantic)."""

import json
import logging
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class CustomTokenizer(BaseModel):
    """Tokenizer custom utilisant une approche gloutonne (Greedy Longest Match)."""
    
    vocab_path: str
    vocab: dict[str, int] = Field(default_factory=dict)
    id_to_token: dict[int, str] = Field(default_factory=dict)
    sorted_tokens: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def load_vocab(self) -> 'CustomTokenizer':
        """Charge le vocabulaire et prépare les structures de données post-instanciation."""
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # S'adapte à la structure de vocab.json selon le modèle
                if "model" in data and "vocab" in data["model"]:
                    self.vocab = data["model"]["vocab"]
                else:
                    self.vocab = data
            
            # 1. Dictionnaire inversé pour la méthode decode()
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            
            # 2. Tri des tokens du plus long au plus court pour la méthode encode()
            self.sorted_tokens = sorted(self.vocab.keys(), key=len, reverse=True)
            
            logger.info(f"Tokenizer custom chargé avec {len(self.vocab)} tokens.")
        except Exception as e:
            logger.critical(f"Erreur de chargement du vocabulaire : {e}")
            raise
            
        return self

    def _clean_text(self, text: str) -> str:
        """Transforme les caractères spéciaux du tokenizer en vrais espaces."""
        return text.replace('Ġ', ' ').replace('\u2581', ' ').replace('<0x20>', ' ')

    def decode(self, token_ids: list[int]) -> str:
        """Décode une liste d'IDs en chaîne de caractères lisible."""
        decoded_str = ""
        for t_id in token_ids:
            if t_id in self.id_to_token:
                token_str = self.id_to_token[t_id]
                decoded_str += self._clean_text(token_str)
            else:
                logger.warning(f"ID Inconnu: {t_id}")
        return decoded_str

    def encode(self, text: str) -> list[int]:
        """Encode une chaîne de caractères en IDs (Greedy Longest Match)."""
        token_ids = []
        
        # Remplacement de l'espace classique par le préfixe spécial courant
        text_to_encode = text.replace(' ', 'Ġ')
        
        i = 0
        while i < len(text_to_encode):
            match_found = False
            
            # Cherche la plus grande sous-chaîne existante
            for token in self.sorted_tokens:
                if text_to_encode.startswith(token, i):
                    token_ids.append(self.vocab[token])
                    i += len(token)
                    match_found = True
                    break
            
            # Fallback caractère par caractère si aucun token ne match
            if not match_found:
                fallback_char = text_to_encode[i]
                if fallback_char in self.vocab:
                    token_ids.append(self.vocab[fallback_char])
                i += 1
                
        return token_ids
