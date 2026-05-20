*This project has been created as part of the 42 curriculum by geoffrey.*

# Call Me Maybe - Constrained Decoding Function Calling LLM

## Description
**Call Me Maybe** is a high-reliability function-calling orchestrator engineered around a highly compact Large Language Model (`Qwen/Qwen3-0.6B`). In modern artificial intelligence systems, Large Language Models (LLMs) excel at processing natural language queries but are inherently prone to syntax hallucinations when required to output strict computer-readable structures. While frontier commercial models rely on sheer parameter volume to maintain formatting, smaller open-source models under 1 billion parameters typically fail to output structured JSON consistently, dropping below a 30% success rate under native conditions.

This project bridges that gap by implementing a rigid token-level **Constrained Decoding** layer driven by a deterministic **Finite State Machine (FSM)**. Instead of attempting to influence the model via fragile prompt engineering patterns, this system directly intercepts the model's token distribution pipeline at every single step of the auto-regressive generation process. By overriding the raw mathematical probabilities (`logits`) before a token is sampled, this application forces the language engine to act as a structured parser, guaranteeing **100% syntactically valid JSON output** that perfectly conforms to a predefined schema under all circumstances.

---

## Instructions

### Prerequisites
* **Operating System:** Linux (Ubuntu 24.04 LTS verified) or macOS.
* **Python Runtime:** Python 3.10 or later.
* **Package Management:** `uv` by Astral (required for ultra-fast, isolated project synchronization).
* **Hardware:** A compatible NVIDIA GPU with CUDA drivers installed is highly recommended for optimal throughput.

### Installation
The project setup is entirely automated via the provided `Makefile`. To provision the environment, create the isolated virtual environment, and install all declared dependencies, run:

```bash
make install
```

*Note: This execution implicitly triggers `uv sync`, mapping requirements directly into the local `.venv/` sandbox.*

### Compilation & Linting Execution
To verify that code meets style rules and rigorous static type validation benchmarks required by the evaluation sheet, utilize the automated verification suite:

```bash
# Runs flake8 verification controls alongside a strict type checking query
make lint

# Runs an exhaustive strict mypy compliance validation across all source definitions
make lint-strict
```

### Basic Execution
To process the baseline evaluation suite using automated path configurations, execute the primary execution wrapper:

```bash
make run
```

---

## Example Usage

### Dynamic Pipeline Adjustments via CLI
The system exposes a rich command-line interface allowing developers to dynamically remap schemas, input manifests, output destinations, and underlying compute hardware:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json \
  --device cuda
```

### Passing Parameters through the Makefile Relay
If you prefer running your workflows exclusively through `make`, you can pass custom parameters into the execution layer using the `ARGS` override variable:

```bash
# Forcing execution onto a specific compute backend
make run ARGS="--device cuda"

# Overriding input scopes and operational hardware simultaneously
make run ARGS="--device cpu --input data/input/custom_tests.json"
```

### Input vs. Output Structural Alignment
Given a typical test element inside `function_calling_tests.json`:

```json
{
  "prompt": "What is the sum of 265 and 345?"
}
```

The application interprets the available capabilities inside `functions_definition.json` and crafts a structured entry inside `data/output/function_calling_results.json` matching the absolute format requirement:

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

## Algorithm Explanation

The architectural backbone of the constraint engine is a deterministic **Finite State Machine (FSM)** defined inside `JsonState`. Rather than inspecting the text after generation, the application operates concurrently with the model's neural layers, evaluating state configurations token-by-token.

```text
       [EXPECT_BRACE_OPEN]  --> Forces '{'
                |
     [EXPECT_NAME_KEY_PREFIX] --> Forces '"name": "'
                |
       [EXPECT_NAME_VALUE]   --> Limits vocabulary to valid function names
                |
    [EXPECT_PARAM_KEY_PREFIX] --> Forces '", "parameters": {'
                |
       [EXPECT_PARAM_KEY]    --> Restricts vocabulary to unused property names
                |
      [EXPECT_PARAM_COLON]   --> Forces '": '
                |
      [EXPECT_PARAM_VALUE]   --> Type-checks numbers vs strings dynamically
                |
      [EXPECT_PARAM_NEXT]    --> Decides between ',' (more keys) and '}' (close block)
                |
       [EXPECT_END_BRACE]    --> Forces final global '}'
                |
             [DONE]          --> Halts inference sequence cleanly
```

### The Logit Modification Loop
When the generation loop requests the next token, the following operations are executed:

1. **Logit Harvesting:** The program calls the SDK's `get_logits_from_input_ids` interface. This returns a vector array representing un-normalized logarithmic probability scores (`logits`) for every item in the 151,936-token vocabulary.
2. **Mask Initialization:** An isolation mask array of identical dimensions is prepared, completely pre-populated with negative infinity (`-np.inf`).
3. **FSM State Query:** The algorithm checks `self.current_state` to understand what grammatical character sequence is legal.
4. **Token Qualification Filter:** The engine scans optimized slices of the vocabulary. For each token, it computes a speculative lookup string. If that token sequence perfectly matches or acts as a valid progressive prefix for the expected grammatical target, its original probability score is copied back into the mask array.
5. **Deterministic Argmax Selection:** The filtered mask is pushed into `np.argmax()`. Since all illegal transitions remain locked at `-np.inf`, the model is mathematically incapable of picking a character sequence that violates the JSON layout.

---

## Design Decisions

### 1. Granular Vocabulary Pre-Filtering and Token Chunking Optimization
Iterating over the complete 151,936-token vocabulary inside a Python `for` loop at every single token step introduces an unsustainable computational penalty. To solve this, `JsonConstraint` implements a highly optimized compilation phase executed exactly once during structural setup within a Pydantic `@model_validator(mode='after')` block. The system analyzes the specific names and parameters declared in the current function configuration file and populates dedicated lightweight lookups (`valid_name_tokens`, `valid_key_token`, `valid_number_tokens`). By tracking these fragments ahead of time, the operational loop processing time drops from hundreds of thousands of operations to small iterative evaluations.

### 2. State Machine Recyclability and Memory Caching
In initial prototypes, a fresh instance of the constraint engine was generated for each separate string prompt. This pattern introduced severe disk IO and structural parsing overheads as the heavy JSON vocabulary was re-read and parsed repeatedly. The final architecture shifts instantiation **outside** the tracking loop. The orchestrator loads the vocabulary exactly once upon initialization, exposing a high-speed `.reset()` method that flushes buffers, restores states back to `EXPECT_BRACE_OPEN`, and clears tracked parameter state tables between consecutive execution queries.

### 3. High-Density Compact Prompt Engineering
Because the evaluation SDK lacks a persistent KV-Cache (Key-Value Cache), every new token requires the model to re-evaluate the entire historical text context array from scratch, establishing a heavy quadratic complexity curve ($O(N^2)$). To maximize token generation speed, all verbose instructions, markdown formatting templates, and generic conversational filler were completely eliminated from the prompt payload. The prompt format was compressed into an ultra-lean signature block:

```text
Extract the request into a JSON function call.
Tools:
- fn_add_numbers(a: number, b: number): Add two numbers together and return their sum.
Request: What is the sum of 2 and 3?
JSON:
```

This data compaction pattern keeps the sequence input size highly optimized, preventing exponential latency spikes during multi-token generation runs.

---

## Performance Analysis

### Operational Precision Metrics
* **Syntactic Compliance Rate:** Achieved a perfect **100% structural parsing validity score**. Every single generated output file parses natively via Python's standard `json.loads()` module without raising format exceptions.
* **Routing Accuracy:** Maintained a **95%+ parameter mapping extraction success rate**, demonstrating that the compact context wrapper provides enough signal for a 500M parameter neural net to make accurate semantic routing decisions.

### Performance Benchmarks (Hardware Compute Backends)
Testing verified the massive performance variance between standard generic processors and dedicated parallel matrix compute engines:

| Compute Configuration | Backend Platform | Combined Processing Time | Architectural Root Cause |
| :--- | :--- | :--- | :--- |
| **CPU Processing Mode** | AMD Ryzen 5 4600H | **13 minutes and 28 seconds** | Single-threaded matrix transformations combined with extensive sequential string casting overhead within the baseline execution framework. |
| **GPU Acceleration Mode**| NVIDIA RTX 3050 | **33 seconds** | Highly parallelized matrix math executions orchestrated via the `accelerate` interface, loading weight structures directly into dedicated high-speed VRAM. |

---

## Challenges Faced

### 1. Tokenizer Boundary Aggregation Anomalies ("Mega-Tokens")
**Problem:** When releasing constraint controls inside open text values (such as typing string attributes), the model would use its deep structural knowledge to output composite trailing patterns in a single token chunk (e.g., generating `'"}}\n'` in one single sequence). Standard step-by-step state engines would freeze or desynchronize because they expected these boundary markers to arrive as individual, isolated tokens.
**Solution:** Developed an intrusive split analysis system within the `consume_token` runtime wrapper. When a closing quotation boundary is detected anywhere inside an incoming token chunk, the remaining string trailing data is captured using `token_str.split('"', 1)[1]`. The state machine evaluates this remainder for structural elements, dynamically forwarding the machine state directly to `EXPECT_END_BRACE` or `DONE`.

### 2. Trailing Comma JSON Violations
**Problem:** In many structural contexts, the model would attempt to generate trailing commas after formatting a field value (e.g., producing `{"a": 16,}`). While legal in languages like Python or C, trailing commas are explicitly forbidden in JSON schemas and cause standard parsers to crash.
**Solution:** Wired the `EXPECT_PARAM_VALUE` state logic directly into the state machine's internal dictionary tracking active variables (`self.remaining_params`). If the length of the remaining tracked variables array is exactly equal to one, the comma logit token is explicitly suppressed, and the closing brace token (`}`) is programmatically locked as the only legal option.

### 3. Leading Numeric Signs and Space Traps
**Problem:** In numeric values, the system would initially unlock validation separators as soon as the buffer wasn't empty. However, the model frequently started numbers with a blank space or a minus sign (`-`), leading to broken extractions like `{"a": -}` if followed by a separator.
**Solution:** Re-engineered the numeric validation step inside `contrain_logits` to execute a explicit numerical check (`any(char.isdigit() for char in self.current_buffer)`). Separators are securely locked out until a real digit is inside the accumulation stream.

---

## Testing Strategy

The correctness of the system was validated against a multi-layer testing pipeline:
1. **Pydantic Structural Enforcement:** Source schema configurations are run through rigid typed models (`Parser`, `Function`, `ParameterProperty`) to ensure that input formatting configurations are healthy before text processing.
2. **Edge-Case Syntactic Scenarios:** Tested with inputs containing multi-word target replacements, special regex expressions, negative boundaries, float integers, and singular structural attributes.
3. **Serialization Order Invariant Rules:** Reconstructed output dictionaries using strict explicit structures to ensure that the primary `prompt` property is persistently indexed as the first key, guaranteeing complete alignment with automated testing frameworks.

---

## Resources & AI Usage Disclosure

### Reference Documents
* **Hugging Face Transformers SDK:** Guidance on logit processing matrices and tokenizer token-to-string extraction mapping workflows.
* **PEP 257 & PEP 484 Standards:** Followed to maintain pristine documentation structures, strict clean docstring configurations, and clean typing assertions throughout the source tree.

### AI Collaboration Statement
AI was utilized as an interactive code auditor, architectural consultant, and systems engineering advisor. Specific integration points included:
* Designing the state machine's split-token calculation rules to smoothly handle composite multi-character tokenizer chunks (`'"}}\n'`).
* Auditing logit mask operations to identify edge cases with negative number formats.
* Debugging environment driver pipeline blockages to correctly transition memory structures from slow CPU workflows to high-speed NVIDIA CUDA parallel execution blocks using the `accelerate` orchestration package.