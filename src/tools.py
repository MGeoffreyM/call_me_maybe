import threading
from enum import Enum, auto
from collections import OrderedDict
from typing import Optional, Any
import numpy as np


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


class ThreadSafeLRUCache:
    """Un cache LRU Thread-Safe utilisant uniquement la librairie standard."""
    def __init__(self, capacity: int = 2048):
        self.cache: OrderedDict[tuple[Any], np.ndarray] = OrderedDict()
        self.capacity = capacity
        self.lock = threading.Lock()

    def get(self, key: tuple[Any]) -> Optional[np.ndarray]:
        """Récupère un masque s'il existe et le marque comme récemment
        utilisé."""
        with self.lock:
            if key not in self.cache:
                return None
            # Hit : On déplace l'élément à la fin (côté MRU)
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: tuple[Any], value: np.ndarray) -> None:
        """Ajoute un masque au cache, et supprime le plus vieux si la limite
        est atteinte."""
        with self.lock:
            self.cache[key] = value
            self.cache.move_to_end(key)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
