"""Point d'entrée principal du programme Call Me Maybe.

Gère l'initialisation du modèle LLM, la boucle de génération sous contrainte,
et la sauvegarde des appels de fonctions au format JSON attendu.
"""

import argparse
import json
import sys
import os
import logging
import concurrent.futures
from typing import cast, Any
import numpy as np
from llm_sdk import Small_LLM_Model
from .log_config import setup_logging, AppLogger
from .parser import Parser, Function, Prompt
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
        args: list[str] = []
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
    profile = MODEL_PROFILES.get(model_name, MODEL_PROFILES["Qwen/Qwen3-0.6B"])

    tools_str = functions_formatter(available_functions)

    full_prompt = profile["prompt_template"].format(
        tools=tools_str,
        prompt=user_prompt
    )

    return full_prompt


def process_single_prompt(prompt_obj: Prompt,
                          prompt_index: int,
                          total_prompts: int,
                          model: Small_LLM_Model,
                          allowed_functions: list[Function],
                          model_name: str,
                          vocab_path: str) -> dict[str, Any] | None:
    """Traite un seul prompt de bout en bout (multi-thread)."""
    prompt_text = prompt_obj.prompt
    prod_logger.info(f"Traitement du prompt [{prompt_index}/{total_prompts}]: "
                     f"'{prompt_text}'")

    constrainer = JsonConstraint(
        vocab_path=vocab_path,
        allowed_functions=allowed_functions,
    )

    full_prompt = build_system_prompt(prompt_text,
                                      allowed_functions,
                                      model_name)

    input_tensor = model.encode(full_prompt)
    input_ids: list[int] = input_tensor[0].tolist()
    # input_ids: list[int] = tokenizer.encode(full_prompt)

    generated_json = ""
    max_tokens = 150
    tokens_generated = 0

    while (constrainer.current_state != JsonState.DONE
           and tokens_generated < max_tokens):

        logits_tensor = model.get_logits_from_input_ids(input_ids)
        logits = np.array(logits_tensor)

        logits_contraints = constrainer.constrain_logits(logits)
        next_token_id = int(np.argmax(logits_contraints))
        token_str = model.decode([next_token_id])
        # token_str = tokenizer.decode([next_token_id])

        input_ids.append(next_token_id)

        generated_json += token_str
        tokens_generated += 1
        constrainer.consume_token(next_token_id, token_str)

        prod_logger.token(f"[P-{prompt_index}] Token: {repr(token_str):8} "
                          f"| État FSM: {constrainer.current_state.name} | "
                          f"Buffer: {repr(constrainer.current_buffer)}")

    # 2. Couper tout ce qui dépasse la dernière accolade fermante
    last_brace = generated_json.rfind('}')
    if last_brace != -1:
        generated_json = generated_json[:last_brace+1]

    try:
        parsed_json = json.loads(generated_json)

        if not isinstance(parsed_json, dict):
            raise ValueError("La sortie JSON n'est pas un dictionnaire.")

        ordered_json = {
            "prompt": prompt_text,
            "name": parsed_json.get("name"),
            "parameters": parsed_json.get("parameters")
        }
        prod_logger.info(f"[P-{prompt_index}] Validé. {tokens_generated} Tokens générés")
        return ordered_json

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Erreur [Prompt {prompt_index}]: JSON invalide "
                     f"({e}):\n{generated_json}")
        return None


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
    #  my_tokenizer = CustomTokenizer(vocab_path=vocab_path)

    results = []

    total_prompts = len(parser.list_prompt)

    max_workers = 4  #  min(6, os.cpu_count() or 4)
    logger.info(f"Lancement de la génération avec {max_workers} "
                "threads simultanés...")

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers) as executor:
        # Préparation des arguments pour chaque prompt
        tasks_args = [
            (prompt_obj,
             i + 1,
             total_prompts,
             model,
             parser.list_function,
             args.model,
             vocab_path) for i, prompt_obj in enumerate(parser.list_prompt)
        ]

        # executor.map exécute en parallèle mais renvoie les résultats dans
        # l'ordre EXACT des entrées
        try:
            # L'utilisation d'une lambda permet de déballer les arguments
            raw_results = list(executor.map(lambda p: process_single_prompt(*p), tasks_args))

            # On filtre les résultats qui ont échoué (None)
            results = [res for res in raw_results if res is not None]

        except Exception as e:
            logger.error(f"Erreur fatale lors du multithreading : {e}")
    try:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"{len(results)}/{total_prompts} résultats sauvegardés "
                    f"dans {args.output}")
    except IOError as e:
        logger.error(f"Erreur d'écriture dans le fichier de sortie: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
