"""Point d'entrée principal du programme Call Me Maybe.

Gère l'initialisation du modèle LLM, l'orchestration du pipeline de génération
de bout en bout, la distribution multithreadée des requêtes,
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
from .tokenizer import CustomTokenizer  # tokenizer perso


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
                          vocab_path: str,
                          tokenizer: CustomTokenizer | None = None) -> dict[
                              str, Any] | None:
    """Process de traitement d'un prompt."""
    prompt_text = prompt_obj.prompt
    prod_logger.info(f"Traitement du prompt [{prompt_index}/{total_prompts}]:"
                     f" '{prompt_text}'")

    constrainer = JsonConstraint(
        vocab_path=vocab_path,
        allowed_functions=allowed_functions,
    )

    full_prompt = build_system_prompt(prompt_text,
                                      allowed_functions,
                                      model_name)

    if tokenizer is not None:
        input_ids = tokenizer.encode(full_prompt)
    else:
        input_tensor = model.encode(full_prompt)
        input_ids = input_tensor[0].tolist()

    generated_json = ""
    max_tokens = 150
    tokens_generated = 0

    while (constrainer.current_state != JsonState.DONE
           and tokens_generated < max_tokens):

        logits_tensor = model.get_logits_from_input_ids(input_ids)
        logits = np.array(logits_tensor)

        logits_contraints = constrainer.constrain_logits(logits)
        next_token_id = int(np.argmax(logits_contraints))

        if tokenizer is not None:
            token_str = tokenizer.decode([next_token_id])
        else:
            token_str = model.decode([next_token_id])

        input_ids.append(next_token_id)

        generated_json += token_str
        tokens_generated += 1
        constrainer.consume_token(next_token_id, token_str)

        prod_logger.token(f"[P-{prompt_index}] Token: {repr(token_str):8} "
                          f"| État FSM: {constrainer.current_state.name} | "
                          f"Buffer: {repr(constrainer.current_buffer)}")

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
        prod_logger.info(f"[P-{prompt_index}] Validé. {tokens_generated}"
                         " Tokens générés")
        return ordered_json

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Erreur [Prompt {prompt_index}]: JSON invalide "
                     f"({e}):\n{generated_json}")
        return None


def main() -> None:
    """Fonction principale d'exécution du pipeline de bout en bout."""
    setup_logging(console_level=logging.INFO)

    # Gestion des différentes options.
    cli_parser = argparse.ArgumentParser(
        description='Call Me Maybe - Function Calling LLM')

    # Options de fichiers.
    cli_parser.add_argument('--functions_definition', type=str,
                            default='data/input/functions_definition.json')
    cli_parser.add_argument('--input', type=str,
                            default='data/input/function_calling_tests.json')
    cli_parser.add_argument(
        '--output', type=str,
        default='data/output/function_calling_results.json')

    # Option de choix du modèle.
    cli_parser.add_argument(
        '--model', type=str, default='Qwen/Qwen3-0.6B',
        help="HuggingFace model ID (ex: TinyLlama/TinyLlama-1.1B-Chat-v1.0)")

    # Option du choix du tokenizer
    cli_parser.add_argument('--custom-tokenizer', action='store_true',
                            help="Utilise le tokenizer custom (Greedy)"
                                 "au lieu du SDK.")

    # Option du choix matériel
    cli_parser.add_argument(
        '--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'],
        help="Forcer le matériel de calcul (cpu, cuda, mps).")

    args = cli_parser.parse_args()

    model = Small_LLM_Model(model_name=args.model,
                            device=args.device)

    if args.custom_tokenizer:
        mode_calcul = "⚠️ Custom Tokenizer (Précision potentiellement réduite)"
    else:
        mode_calcul = "SDK Tokenizer"

    logger.info(f"Modèle chargé : {args.model}")
    logger.info(f"Mode du tokenizer : {mode_calcul}")

    parser = Parser()
    parser.read_files(args.functions_definition, args.input)

    vocab_path = model.get_path_to_tokenizer_file()
    my_tokenizer = (
        CustomTokenizer(vocab_path=vocab_path) if args.custom_tokenizer
        else None
    )

    results = []

    total_prompts = len(parser.list_prompt)

    max_workers = 4
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
             vocab_path,
             my_tokenizer) for i, prompt_obj in enumerate(parser.list_prompt)
        ]

        # executor.map exécute en parallèle mais renvoie les résultats dans
        # l'ordre EXACT des entrées
        try:
            raw_results = list(
                executor.map(lambda p: process_single_prompt(*p), tasks_args))

            results = [res for res in raw_results if res is not None]

        except Exception as e:
            logger.error(f"Erreur fatale lors du multithreading : {e}")

    # génération du fichier de sortie.
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
