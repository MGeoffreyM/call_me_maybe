"""Point d'entrée principal du programme Call Me Maybe.

Gère l'initialisation du modèle LLM, la boucle de génération sous contrainte,
et la sauvegarde des appels de fonctions au format JSON attendu.
"""

import argparse
import json
import sys
import os
import logging
from typing import cast
import numpy as np
from llm_sdk import Small_LLM_Model
from .log_config import setup_logging, AppLogger
from .parser import Parser, Function
from .constrainer import JsonConstraint, JsonState


logger = logging.getLogger(__name__)
prod_logger = cast(AppLogger, logging.getLogger("production"))

MODEL_PROFILES = {
    "Qwen/Qwen3-0.6B": {
        "prompt_template": "Extract the request into a JSON function call."
                           "\nTools:\n{tools}\nRequest: {prompt}\nJSON:\n"
    },
    "bigscience/bloomz-560m": {
        "prompt_template": "Instruction: Extract the user request into a JSON"
                           " function call using these tools.\n\nTools:\n"
                           "{tools}\n\nUser request: {prompt}\n\nJSON output:"
                           "\n"
    },
    "HuggingFaceTB/SmolLM2-360M-Instruct": {
        "prompt_template": "<|im_start|>system\nYou are a helpful assistant. "
                           "Extract"" the user request into a JSON function "
                           "call.\n\nTools:\n{tools}<|im_end|>\n<|im_start|>"
                           "user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    }
}


def functions_formatter(functions: list[Function]) -> str:
    """Formate la liste des fonctions disponibles pour le prompt système.

    Args:
        functions (list[Function]): La liste des fonctions parsées.

    Returns:
        str: Une chaîne de texte décrivant les fonctions et leurs arguments.
    """
    texte = ""
    for func in functions:
        args = []
        if func.parameters:
            for p_name, p_prop in func.parameters.items():
                args.append(f"{p_name}: {p_prop.type}")
        args_str = ", ".join(args)

        texte += f"- {func.name}({args_str}): {func.description}\n"

    return texte


def build_system_prompt(user_prompt: str,
                        available_functions: list[Function],
                        model_name: str = "Qwen/Qwen3-0.6B") -> str:
    """Construit le prompt textuel adapté au modèle choisi.

    Args:
        user_prompt (str): La requête initiale en langage naturel.
        available_functions (list[Function]): Liste des fonctions autorisées.
        model_name (str): Le nom du modèle en cours d'utilisation.

    Returns:
        str: Le prompt formaté intégrant les instructions et outils
        spécifiques.
    """
    # Récupère le profil du modèle ou se rabat sur Qwen par défaut
    profile = MODEL_PROFILES.get(model_name, MODEL_PROFILES["Qwen/Qwen3-0.6B"])

    tools_str = functions_formatter(available_functions)

    # Injecte dynamiquement les variables dans le template
    full_prompt = profile["prompt_template"].format(
        tools=tools_str,
        prompt=user_prompt
    )

    return full_prompt


def main() -> None:
    """Fonction principale d'exécution du pipeline de bout en bout."""

    setup_logging(console_level=logging.INFO)

    cli_parser = argparse.ArgumentParser(
        description='Call Me Maybe - Function Calling LLM')
    cli_parser.add_argument('--functions_definition', type=str,
                            default='data/input/functions_definition.json')
    cli_parser.add_argument('--input', type=str,
                            default='data/input/function_calling_tests.json')
    cli_parser.add_argument(
        '--output', type=str,
        default='data/output/function_calling_results.json')
    cli_parser.add_argument(
        '--model', type=str, default='Qwen/Qwen3-0.6B',
        help="HuggingFace model ID (ex: TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    cli_parser.add_argument(
        '--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'],
        help="Forcer le matériel de calcul (cpu, cuda, mps).")

    args = cli_parser.parse_args()

    model = Small_LLM_Model(model_name=args.model,
                            device=args.device)
    mode_calcul = args.device.upper() if args.device else "AUTOMATIQUE"

    logger.info(f"Modèle chargé : {args.model}")
    logger.info(f"Matériel utilisé pour l'IA : {mode_calcul}")

    parser = Parser()
    parser.read_files(args.functions_definition, args.input)

    vocab_path = model.get_path_to_tokenizer_file()
    results = []

    constrainer = JsonConstraint(
        vocab_path=vocab_path,
        allowed_functions=parser.list_function,
    )

    for prompt_obj in parser.list_prompt:
        prompt_text = prompt_obj.prompt
        prod_logger.info(f"Traitement du prompt : '{prompt_text}'")

        constrainer.reset()
        input_tensor = model.encode(
            build_system_prompt(prompt_text,
                                constrainer.allowed_functions,
                                args.model))
        input_ids = input_tensor[0].tolist()

        generated_json = ""
        max_tokens = 150
        tokens_generated = 0

        while (constrainer.current_state != JsonState.DONE
               and tokens_generated < max_tokens):

            # A. Obtenir les logits bruts
            logits_tensor = model.get_logits_from_input_ids(input_ids)
            logits = np.array(logits_tensor)

            # B. Filtrer les logits avec le masque restrictif de la FSM
            logits_contraints = constrainer.constrain_logits(logits)

            # C. Sélectionner le token vainqueur
            next_token_id = int(np.argmax(logits_contraints))

            # D. Décoder *uniquement* le nouveau token
            token_str = model.decode([next_token_id])

            # E. Mises à jour pour le prochain tour
            input_ids.append(next_token_id)
            generated_json += token_str
            tokens_generated += 1

            # F. Informer la Machine à États du token choisi !
            constrainer.consume_token(next_token_id, token_str)

            prod_logger.token(f"Token: {repr(token_str):8} | État FSM: "
                              f"{constrainer.current_state.name} | "
                              f"Buffer: {repr(constrainer.current_buffer)}")

        # Construction de l'objet de sortie final pour ce prompt
        try:
            parsed_json = json.loads(generated_json)

            if not isinstance(parsed_json, dict):
                raise ValueError("La sortie JSON n'est pas un dictionnaire.")

            ordered_json = {
                "prompt": prompt_text,
                "name": parsed_json.get("name"),
                "parameters": parsed_json.get("parameters")
            }
            results.append(ordered_json)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Erreur : JSON invalide ({e}):\n{generated_json}")
            continue

    try:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"{len(results)} résultats sauvegardés dans "
                    f"{args.output}")
    except IOError as e:
        logger.error(f"Erreur d'écriture dans le fichier de sortie: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
