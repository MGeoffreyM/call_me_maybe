*This project has been created as part of the 42 curriculum by gematura.*

# Call Me Maybe - Function Calling LLM via Constrained Decoding

**🇫🇷 [Version Française](README-fr.md)**

## Description
**Call Me Maybe** is a highly reliable function-calling orchestrator built around compact Small Language Models (SLMs) such as `Qwen/Qwen3-0.6B`.

In modern AI systems, Large Language Models (LLMs) excel at processing natural language queries but are inherently prone to syntax hallucinations when asked to output strict data structures. While state-of-the-art commercial models rely on massive parameter counts to maintain formatting, smaller models (under 1 billion parameters) generally fail to generate consistent structured JSON.

This project bridges that gap by implementing a rigid layer of **Constrained Decoding** at the token level, driven by a deterministic **Finite State Machine (FSM)**. By overriding the raw mathematical probabilities (`logits`) before a token is sampled, this application forces the language engine to act as a structured parser, guaranteeing **syntactically valid JSON output** that perfectly conforms to a predefined schema under all circumstances.

---

## Instructions

### Prerequisites
* **Operating System:** Linux or macOS.
* **Python Environment:** Python 3.10 or higher.
* **Package Manager:** `uv` (An ultra-fast Python package installer and resolver written in Rust, used for seamless dependency tracking and virtual environment isolation).

### Installation
Create the virtual environment and install dependencies:
```bash
make install
```

### Usage
* **Standard Execution:**
  ```bash
  # Default run: Qwen/Qwen3-0.6B model using the native SDK tokenizer
  make run
  ```
* **Selecting a Specific Language Model:**
  ```bash
  make run bloomz      # Uses the bigscience/bloomz-560m model
  make run smollm2     # Uses the HuggingFaceTB/SmolLM2-360M-Instruct model
  ```
* **Using the Custom Tokenizer:**
  Run Qwen with the custom Greedy Longest Match tokenizer:
  ```bash
  make run custom
  ```
* **Combining Model and Tokenizer Modifiers:**
  ```bash
  make run smollm2 custom
  ```
* **Passing Custom Paths or Flags:**
  ```bash
  # Example:
  make run ARGS="--input custom_prompts.json --output custom_results.json"

  # Available flags:
    --functions_definition <path>
    --input <path>
    --output <path>
    --device <cpu|cuda|mps>
  ```

* **Running the Test Suite:**
  The engine integrates a comprehensive testing pipeline that supports isolated or combined execution of test suites, coupled with any target model or tokenizer.
  ```bash
  # Run ALL test suites across all categories (default)
  make test

  # Run targeted test suites
  make test extrem       # Runs edge cases, large numbers, and extreme boundaries
  make test stress       # Runs prompt distribution tests under high load
  make test complex      # Runs deep nested object function structures

  # Cross-testing with models and the custom tokenizer bonus
  make test complex custom
  make test stress smollm2
  make test extrem bloomz custom
  ```

* **Debugging:**
  Launch the program with `pdb`:
  ```bash
  make debug  
  ```

* **Code Quality & Linting:**
  ```bash
  make lint          # Runs flake8 and mypy checks
  make lint-strict   # Runs strict flake8 and mypy validation
  ```

* **Cleanup:**
  ```bash
  make clean   # Removes temporary files and output results.
  make fclean  # Executes clean and completely removes the virtual environment.
  ```

---

## Algorithms, Decisions, and Performance

### Algorithm
* **FSM (Finite State Machine):** The architectural backbone is a deterministic **Finite State Machine** defined in `JsonState`. The application runs concurrently with the model's neural layers, evaluating state configurations on a token-by-token basis.

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
      [EXPECT_PARAM_VALUE]      --> Dynamically type-checks numbers, strings, and booleans
                |
      [EXPECT_PARAM_NEXT]       --> Decides between ',' (more keys) or '}' (close block)
```

* **The Logit Modification Loop:**
1. The program calls `get_logits_from_input_ids` to retrieve the raw probability scores (`logits`) for the entire vocabulary.
2. An isolation mask array is prepared, pre-filled with negative infinity (`-np.inf`).
3. Based on `self.current_state` and `self.current_buffer`, the algorithm identifies legal tokens (e.g., prefix matching for `"true"` or `"false"` in boolean mode).
4. The original logits of the valid tokens are copied into the mask.
5. `np.argmax()` selects the best token. Since all illegal transitions remain at `-np.inf`, the model is mathematically incapable of violating the JSON structure.

### Design Decisions
* **Context Stack for Recursive Structure:** To support multi-depth nesting, the FSM uses an active tracking list (`context_stack`). When an `object` parameter is encountered and its schema contains sub-properties, a new sub-context dictionary is pushed onto the stack. Closing braces (`}`) dynamically pop the current context to seamlessly return to parent scopes.
* **Multi-thread Processing:** To accelerate the sequential inference imposed by a batch size of 1, we implemented multithreaded parallel processing (`concurrent.futures`).
* **Advanced Logging:** A custom logging system features ANSI color codes, separation of production (`prod_call_me_maybe.log`) and system (`sys_call_me_maybe.log`) logs, along with a specific trace level to monitor token-by-token generation.
* **Memory Optimization via Token Caching:** Logit constraint masks are heavy arrays to build on the fly. We designed a static cache table (`_mem_cache`) indexing the combinations of `(current_state, current_buffer, remaining_keys)`. This avoids reallocating masks during simultaneous execution contexts, massively accelerating decoding speed.

### Performance Analysis
Our architecture was designed to balance strict structural validity with inference time, bypassing the inherent limits of token-by-token processing on small models.

* **Reliability and Accuracy:** Without constraints, a 0.5B parameter model fails to generate valid JSON in a vast majority of cases. With our dynamic FSM, syntactic validity and schema compliance reach **100%**. Every generated key strictly matches the provided definitions.
* **Computational Overhead:** Evaluating the syntax tree (FSM) theoretically introduces a penalty at each generated token. However, thanks to vocabulary pre-filtering during instantiation (restricted lists) and the `_mem_cache` system, the logit masking operation executes in **$O(1)$** time in the main loop. Generation speed remains almost exclusively dictated by the LLM's forward pass time.
* **Memory Footprint (RAM vs. CPU Time Trade-off):** Masking requires manipulating NumPy arrays proportional to the vocabulary size (~150,000 floats per token). Rather than reallocating these heavy structures on the fly, the `_mem_cache` stores the computed masks by state. This consumes a negligible amount of additional RAM while saving significant CPU cycles on long generation streams.
* **Scalability, Multithreading, and GIL (Global Interpreter Lock) Management:** Constrained decoding requires using a *Batch Size of 1* at the LLM level, as each prompt follows a unique and unpredictable FSM state path. To process input files massively without suffering from this linearity, the pipeline distributes queries via a `ThreadPoolExecutor` configured to `max_workers=4`. This number represents the perfect hardware and software compromise:
    1. *GPU Protection:* It aligns with the available video memory (VRAM) bandwidth, avoiding hardware saturation or *Out Of Memory* (OOM) errors that overly aggressive concurrency (e.g., 16 or 32 workers) would cause.
    2. *Real Parallelism and GIL Release:* Although Python has a global lock (GIL) limiting pure code execution to a single CPU core, underlying C++-based libraries (like PyTorch during the model's forward pass or NumPy during tensor calculations) **explicitly release the GIL**. Neural calculations thus run in parallel at full speed on the GPU or available CPU cores.
    3. *GIL Contention (GIL Thrashing) Prevention:* At the end of each forward pass, every thread must reacquire the GIL to execute the semantic logic of our FSM in pure Python (`constrainer.py`). Limiting the application to 4 active threads prevents a massive bottleneck phenomenon (where threads spend more time fighting for the lock than progressing), ensuring maximum fluidity. Furthermore, this leaves the test machine's remaining cores free for operating system stability.

Thanks to this optimized parallelization, the entire test suite (including Stress Tests) is processed well under the strict 5-minute limit.

### Challenges Faced

**BPE Tokenization vs. Greedy Match Semantic Loss**
* **Challenge:** To remove direct dependencies on the SDK, we developed a public `CustomTokenizer` using a *Greedy Longest Match* algorithm. While structurally perfect, it revealed a significant challenge: LLM neural pathways are tied to specific token boundaries generated by their native **Byte-Pair Encoding (BPE)** slicing during training.
* **Impact:** Text segments like `"schrek"` were sliced into different sub-words than those expected by the model, leading to semantic degradation (e.g., the model hallucinating `"frodo"` instead of extracting the requested word).
* **Resolution:** We implemented a secure architectural separation. The application defaults to high-fidelity native BPE decoding, while the custom Greedy tokenizer remains fully isolable via the `--custom-tokenizer` CLI flag and the `make run custom` command for educational evaluation of the bonus.

**Tokenizer Boundary Aggregation Anomalies ("Mega-Tokens")**
* **Issue:** The model occasionally produces composite end patterns in a single token block (e.g., `'"}}\n'`). Standard FSMs crash because they expect these boundary markers to be isolated tokens.
* **Solution:** Developed a split-parsing system. When a closing boundary is detected inside an incoming token, the remainder is dynamically evaluated, shifting the state directly to `EXPECT_END_BRACE` or `DONE`.

**JSON Violations Due to Trailing Commas**
* **Issue:** The model attempted to generate trailing commas (`{"a": 16,}`).
* **Solution:** Hooked the `EXPECT_PARAM_VALUE` state logic directly into the active variable tracker. If only one expected parameter remains, the comma token is explicitly suppressed, and `}` is locked as the only legal option.

---

## Implemented Bonus Features

This project successfully implements the following bonus features:

1. **Support for multiple LLM models:** Integrated `MODEL_PROFILES` with custom prompt templates to dynamically support `bigscience/bloomz-560m` and `HuggingFaceTB/SmolLM2-360M-Instruct`.
2. **Comprehensive test suite:** Developed extreme prompts, stress tests, complex nested object tests, and format validation tests (fully automated via `make test`).
3. **Visualization of the generation process:** Implemented an `AppLogger` with ANSI color coding and a custom `TOKEN_LEVEL` log level to visualize the FSM state, buffer, and exact generated token in real-time.
4. **Support for complex nested function arguments:** Implemented a context stack (`self.push_context()` / `self.pop_context()`) within the FSM to support recursive object tracking and nested dictionary generation.
5. **Performance optimizations (caching, batching):**
    * *Caching (CPU Optimization):* Logit mask computation is heavily optimized by caching each generated mask according to the exact FSM state, allowing $O(1)$ retrieval from shared memory.
    * *Batching/Parallelization (I/O Optimization):* The engine distributes tasks across 4 simultaneous threads, drastically reducing the overall test suite execution time.
6. **Advanced error recovery mechanisms:** Generation errors (e.g., `JSONDecodeError`) are individually intercepted in `process_single_prompt()` and isolated, allowing the rest of the multithreaded batch to finish without crashing. Diagnostics are provided by a robust logging system with fail-safe file rotation.
7. **Recoding the tokenizer &**
8. **Public implementation of tokenizer encode and decode methods:** The official SDK tokenizer was completely recoded. It independently loads the JSON vocabulary and exposes two public methods: `encode(text)` and `decode(token_ids)` using a Greedy Longest Match algorithm.
9. **Demonstration of encoding/decoding integration with constrained decoding:** Integration is demonstrated via the `--custom-tokenizer` flag. When active, the pipeline bypasses the model's SDK entirely for text handling, mapping text to tensor inputs via `CustomTokenizer.encode()`, and reconverting output token IDs via `CustomTokenizer.decode()` before feeding them to the FSM.

---

## Resources

### Reference Documents
* **Hugging Face:** The standard platform for Open Source models, Tokenizers, and the Transformers ecosystem. [https://huggingface.co/](https://huggingface.co/)
* Official PyTorch documentation.
* Pydantic documentation for Python data validation, recursive class modeling, and runtime reconstruction (`model_rebuild`).
* POSIX Threads & GIL documentation for conceptual modeling of multithreaded application performance under standard interpreter conditions.

### AI Usage
During this project, Artificial Intelligence was used in a targeted manner as a technical assistant:
* **Learning and comprehension:** Assimilating new concepts (such as latent space, logits, and Constrained Decoding).
* **Translation:** Translating and synthesizing various technical papers and library documentation.
* **Guidance and brainstorming:** Reflecting on the Finite State Machine (FSM) architecture and complex tree-traversal logic.
* **Debugging:** Assisting in identifying and resolving bugs encountered during development (strict typing management, unexpected multithreading behaviors with the Python GIL).
* **Readme:** Drafting and structuring this README to accurately document the complex behaviors of the FSM and the implemented bonus features.
