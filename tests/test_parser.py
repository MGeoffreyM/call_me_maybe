import pytest
from src.parser import Parser


def test_parser_valid_files() -> None:
    """Vérifie que le parser charge correctement des JSON valides statiques."""
    parser = Parser()

    # Lecture des fichiers
    parser.read_files(
        "tests/fixtures/valid_funcs.json",
        "tests/fixtures/valid_prompts.json"
    )

    # Vérification que Pydantic a bien lu les 3 fonctions
    assert len(parser.list_function) == 3

    # Vérification des types classiques
    assert parser.list_function[0].name == "fn_add_numbers"
    assert parser.list_function[1].parameters["name"].type == "string"

    # Vérification des nouveaux types (Bonus robustesse)
    assert parser.list_function[2].name == "fn_create_user"
    assert parser.list_function[2].parameters["is_admin"].type == "boolean"
    assert parser.list_function[2].parameters["roles"].type == "array"

    # Vérification des prompts
    assert len(parser.list_prompt) == 3
    assert parser.list_prompt[2].prompt.startswith("Create an admin user")


def test_parser_invalid_json() -> None:
    """Vérifie que le programme exit(1) proprement sur un mauvais JSON."""
    parser = Parser()

    # On utilise pytest.raises pour intercepter le sys.exit(1)
    with pytest.raises(SystemExit) as excinfo:
        parser.read_files(
            "tests/fixtures/invalid.json",
            "tests/fixtures/valid_prompts.json"
        )

    # Vérification que le code de sortie est bien 1
    assert excinfo.value.code == 1


def test_parser_missing_file() -> None:
    """Vérifie que le programme exit(1) si le fichier n'existe pas."""
    parser = Parser()

    with pytest.raises(SystemExit):
        parser.read_files(
            "tests/fixtures/fichier_fantome.json",
            "tests/fixtures/valid_prompts.json"
        )
