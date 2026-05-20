import pytest
import json
import numpy as np
from pathlib import Path
from src.constrainer import JsonConstraint, JsonState
from src.parser import Function, ParameterProperty


@pytest.fixture
def dummy_vocab_file(tmp_path: Path) -> str:
    """Crée un faux vocabulaire minimaliste pour les tests."""
    vocab = {
        "{": 0,
        "\"name\": \"": 1,
        "fn_test": 2,
        "\"": 3,
        ", \"parameters\": {": 4,
        "}": 5,
        "\"age\"": 6,
        ": ": 7,
        "42": 8
    }
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps(vocab))
    return str(vocab_file)


@pytest.fixture
def dummy_functions() -> list[Function]:
    """Crée une fausse définition de fonction."""
    return [
        Function(
            name="fn_test",
            description="Une fonction de test",
            parameters={"age": ParameterProperty(type="number")},
            returns=ParameterProperty(type="number")
        )
    ]


def test_initial_state(dummy_vocab_file: str,
                       dummy_functions: list[Function]) -> None:
    """Vérifie que la FSM démarre bien en attendant une accolade {."""
    constrainer = JsonConstraint(
        vocab_path=dummy_vocab_file,
        allowed_functions=dummy_functions
    )
    assert constrainer.current_state == JsonState.EXPECT_BRACE_OPEN


def test_constrain_logits_blocks_invalid_tokens(dummy_vocab_file: str,
                                                dummy_functions: list[Function]
                                                ) -> None:
    """Vérifie que la FSM bloque les mauvais tokens au démarrage."""
    constrainer = JsonConstraint(
        vocab_path=dummy_vocab_file,
        allowed_functions=dummy_functions
    )

    # On simule un tableau de logits bruts venant du LLM (que des 1.0)
    raw_logits = np.ones(9)

    # On applique le masque de contrainte
    constrained_logits = constrainer.constrain_logits(raw_logits)

    # Le token 0 est "{". Il doit être autorisé (donc rester à 1.0)
    assert constrained_logits[0] == 1.0

    # Le token 2 est "fn_test". Il n'a rien à faire ici au début !
    # Il doit être bloqué.
    assert constrained_logits[2] == -np.inf


def test_state_transition_consume_token(dummy_vocab_file: str,
                                        dummy_functions: list[Function]
                                        ) -> None:
    """Vérifie le token '{' fait avancer la machine à l'état suivant."""
    constrainer = JsonConstraint(
        vocab_path=dummy_vocab_file,
        allowed_functions=dummy_functions
    )

    # La FSM "mange" le token "{"
    constrainer.consume_token(token_id=0, token_str="{")

    # Elle doit maintenant attendre la clé "name"
    assert constrainer.current_state == JsonState.EXPECT_NAME_KEY_PREFIX


def test_reset_function(dummy_vocab_file: str,
                        dummy_functions: list[Function]) -> None:
    """Vérifie que la fonction reset remet à zéro pour le prochain prompt."""
    constrainer = JsonConstraint(
        vocab_path=dummy_vocab_file,
        allowed_functions=dummy_functions
    )

    constrainer.current_state = JsonState.DONE
    constrainer.current_buffer = "buffer plein"

    constrainer.reset()

    assert constrainer.current_state == JsonState.EXPECT_BRACE_OPEN
    assert constrainer.current_buffer == ""
