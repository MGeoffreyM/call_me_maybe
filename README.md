*This project has been created as part of the 42 curriculum by geoffrey.*

# Call Me Maybe - Constrained Decoding Function Calling LLM

🇫🇷 [Version Française](fr_README_fr.md)

## Description
**Call Me Maybe** is a high-reliability function-calling orchestrator engineered around compact Large Language Models (default: `Qwen/Qwen3-0.6B`). In modern artificial intelligence systems, Large Language Models (LLMs) excel at processing natural language queries but are inherently prone to syntax hallucinations when required to output strict computer-readable structures. While frontier commercial models rely on sheer parameter volume to maintain formatting, smaller models under 1 billion parameters typically fail to output structured JSON consistently.

This project bridges that gap by implementing a rigid token-level **Constrained Decoding** layer driven by a deterministic **Finite State Machine (FSM)**. By overriding the raw mathematical probabilities (`logits`) before a token is sampled, this application forces the language engine to act as a structured parser, guaranteeing **100% syntactically valid JSON output** that perfectly conforms to a predefined schema under all circumstances.

---

## Instructions

### Prerequisites
* **Operating System:** Linux or macOS.
* **Python Runtime:** Python 3.10 or later.
* **Package Management:** `uv` (required for ultra-fast, isolated project synchronization).

### Installation
The project setup is entirely automated via the provided `Makefile`. To provision the environment, create the isolated virtual environment, and install all declared dependencies, run:

```bash
make install
```

### Execution & Make Rules
The project includes a robust `Makefile` for streamlined execution.

**Standard Execution (Qwen3-0.6B):**
```bash
make run
```

**Run with alternative Models (Bonus):**
```bash
make run-bloomz    # Runs with bigscience/bloomz-560m
make run-smollm2   # Runs with HuggingFaceTB/SmolLM2-360M-Instruct
```

**Run the Comprehensive Test Suite (Bonus):**
```bash
make test
```

**Code Quality & Linting:**
```bash
make lint          # Runs flake8 and mypy checks
make lint-strict   # Runs strict mypy validation
```

---

## Example Usage

### Dynamic Pipeline Adjustments via CLI
The system exposes a CLI allowing developers to dynamically remap schemas, input manifests, output destinations, and compute hardware, strictly following the subject's requirements:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json \
  --device cuda
```

### Input vs. Output Structural Alignment
Given a typical test element inside `function_calling_tests.json`:

```json
{
  "prompt": "What is the sum of 265 and 345?"
}
```

The application interprets the available capabilities inside `functions_definition.json` and crafts a structured entry inside the output file:

```json
[
  {
    "prompt": "What is the sum of 265 and 345?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 265.0,
      "b": 345.0
    }
  }
]
```

---

## Bonus Features Implemented

This project successfully implements the following bonus features:

1. **Support for multiple LLM models:** Integrated `MODEL_PROFILES` with custom prompt templates to dynamically support `bigscience/bloomz-560m` and `HuggingFaceTB/SmolLM2-360M-Instruct`.
2. **Comprehensive Test Suite:** Developed extreme prompts, stress tests, complex nested objects tests, and format validation tests (fully automated via `make test`).
3. **Visualization of the Generation Process:** Implemented an `AppLogger` with ANSI color coding and a custom `TOKEN_LEVEL` to visualize the FSM state, the buffer, and the exact token generated in real-time.
4. **Support for Complex Nested Function Arguments:** Implemented a context stack (`self.push_context()` / `self.pop_context()`) inside the FSM to support recursive object tracking and nested dictionary generation.
5. **Advanced Error Recovery Mechanisms:** Added an absolute security "Circuit Breaker" to detect repetitive token generation loops and force token closure to prevent infinite hallucinations.

---

## Algorithm Explanation

The architectural backbone is a deterministic **Finite State Machine (FSM)** defined inside `JsonState`. The application operates concurrently with the model's neural layers, evaluating state configurations token-by-token.

```text
       [EXPECT_BRACE_OPEN]      --> Forces '{'
                |
     [EXPECT_NAME_KEY_PREFIX]   --> Forces '"name":"'
                |
       [EXPECT_NAME_VALUE]      --> Limits vocabulary to valid function names
                |
    [EXPECT_PARAM_KEY_PREFIX]   --> Forces '","parameters":{'
                |
       [EXPECT_PARAM_KEY]       --> Restricts vocabulary to unused property names
                |
      [EXPECT_PARAM_COLON]      --> Forces '":'
                |
      [EXPECT_PARAM_VALUE]      --> Type-checks numbers, strings, and booleans dynamically
                |
      [EXPECT_PARAM_NEXT]       --> Decides between ',' (more keys) or '}' (close block)
```

**The Logit Modification Loop:**
1. The program calls `get_logits_from_input_ids` to retrieve raw probability scores (`logits`) for the entire vocabulary.
2. An isolation mask array is prepared, pre-populated with negative infinity (`-np.inf`).
3. Based on `self.current_state` and `self.current_buffer`, the algorithm identifies legal tokens (e.g., prefix matching for `"true"` or `"false"` in boolean mode).
4. The original logits for valid tokens are copied into the mask.
5. `np.argmax()` selects the best token. Since all illegal transitions remain at `-np.inf`, the model is mathematically incapable of violating the JSON layout.

---

## Design Decisions

### 1. Granular Vocabulary Pre-Filtering
Iterating over the 151,936-token vocabulary for every token generation introduces unsustainable overhead. To solve this, `JsonConstraint` executes a highly optimized compilation phase within a Pydantic `@model_validator`. It populates dedicated lightweight lookups (`valid_name_tokens`, `valid_key_tokens`, `valid_boolean_tokens`), reducing operational loop time drastically.

### 2. Context Stack for Nested Objects
To handle the bonus requirement of complex nested objects, the FSM uses a stack (`context_stack`). When an `object` type is encountered, the FSM pushes the current context and restarts the key/value parsing logic recursively, popping the context only when the nested `}` is successfully generated.

### 3. FSM Recyclability
A fresh instance of the constraint engine is NOT generated for each prompt. Instead, the orchestrator loads the vocabulary exactly once upon initialization and exposes a high-speed `.reset()` method that flushes buffers and context stacks between prompts.

---

## Performance Analysis

* **Syntactic Compliance Rate:** Achieved a **100% structural parsing validity score**. Every output file parses natively via `json.loads()`.
* **Routing Accuracy:** Maintained a **>90% parameter mapping success rate**.
* **Speed:** Due to the vocabulary pre-filtering, inference speed is strictly bounded by the LLM's forward pass, introducing near-zero Python overhead to the generation loop. Hardware acceleration via `--device cuda` reduces inference time from minutes (CPU) to seconds.

---

## Challenges Faced

### 1. Tokenizer Boundary Aggregation Anomalies ("Mega-Tokens")
**Problem:** The model occasionally outputs composite trailing patterns in a single token chunk (e.g., `'"}}\n'`). Standard FSMs freeze because they expect these boundary markers as isolated tokens.
**Solution:** Developed a split analysis system. When a closing boundary is detected inside an incoming token, the remainder is evaluated dynamically, forwarding the state directly to `EXPECT_END_BRACE` or `DONE`.

### 2. The Boolean Hallucination Trap
**Problem:** While waiting for a boolean, if the FSM checks for the exact words `"true"` or `"false"` to validate the input, the LLM is free to output any random string in the meantime, breaking the JSON.
**Solution:** Implemented strict prefix-matching. Logits are mathematically restricted *only* to tokens that start with the letters of "true" or "false" (e.g., 't', 'tr', 'f', 'fa').

### 3. Trailing Comma JSON Violations
**Problem:** The model would attempt to generate trailing commas (`{"a": 16,}`).
**Solution:** Wired the `EXPECT_PARAM_VALUE` state logic directly into the active variables tracker. If only one expected parameter remains, the comma token is explicitly suppressed, and `}` is locked as the only legal option.

---

## Testing Strategy

The correctness of the system was validated against a multi-layer pipeline:
1. **Pydantic Validation:** All input files are strictly validated before processing.
2. **Stress & Extreme Prompts:** Tested with huge numbers, negative floats, special characters, and regex configurations (`data/test_input/test_extrem_prompts.json`).
3. **Invalid Format Handling:** Confirmed that the system gracefully handles broken JSON schemas (`invalid_functions_format.json`) and missing keys via robust `try-except` blocks, never crashing unexpectedly.

---

## Resources & AI Usage Disclosure

### Reference Documents
* **Hugging Face Transformers SDK:** Guidance on logit processing matrices.
* **PEP 257 & PEP 484 Standards:** Followed for clean docstring configurations and strict typing.

### AI Collaboration Statement
AI was utilized as an interactive code auditor and systems engineering advisor. Specific integration points included:
* Refining the state machine's strict boolean prefix-matching logic to guarantee 100% valid schema output.
* Auditing logit mask operations to identify edge cases with negative number formats and trailing commas.
* Structuring this README to properly document complex FSM behaviors and bonus features according to standard developer documentation practices.
