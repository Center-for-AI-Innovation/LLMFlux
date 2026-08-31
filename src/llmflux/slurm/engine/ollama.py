import shlex
import sys
from pathlib import Path

# This function generates an ollama batch script for running llmflux

def create_ollama_batch_script(
    account: str,
    partition: str,
    nodes: str,
    gpus_per_node: str,
    time: str,
    memory: str,
    cpus_per_task: str,
    logs_dir: Path,
    input_file: Path,
    output_file: Path,
    job_name: str,
    slurm_config,
    mode: str = "batch",
    email: str = "",
) -> list[str]:
    # Create SLURM job script
    job_script = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --gpus-per-node={gpus_per_node}",
        f"#SBATCH --time={time}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --output={logs_dir}/%j.out",
        f"#SBATCH --error={logs_dir}/%j.err",
    ]

    # Add extra SBATCH directives if provided
    if slurm_config.extra_sbatch_args:
        for key, value in slurm_config.extra_sbatch_args.items():
            job_script.append(f"#SBATCH --{key}={value}")

    if mode == "serve":
        job_script.extend([
            f"#SBATCH --mail-type=FAIL",
            f"#SBATCH --mail-user={email}",
        ])

    # NOTE: Do NOT `module purge` (or load host gcc/cuda) here. The batch
    # processor further down runs on the *host* and does `import llmflux`, which
    # resolves via the environment that `module load llmflux` put on PATH.
    # Purging it left `python3` as a base interpreter without llmflux, so the
    # processor died with `ModuleNotFoundError: No module named 'llmflux'` while
    # the ollama server (in the container) was otherwise healthy. The container
    # ships its own CUDA, so no host toolchain modules are needed here, and the
    # hardcoded gcc/cuda module names matched nothing on our clusters anyway.
    # vllm.py never purged, which is why the vLLM engine was unaffected.
    job_script.extend([
        "",
        "# Create all necessary directories",
        "mkdir -p $DATA_INPUT_DIR $DATA_OUTPUT_DIR $MODELS_DIR $LOGS_DIR $CONTAINERS_DIR $APPTAINER_TMPDIR $APPTAINER_CACHEDIR",
        "",
        "# Start Ollama server",
        "mkdir -p $OLLAMA_HOME $OLLAMA_MODELS",
        "",
        "# Build container if needed (or if forced)",
        "if [ \"$LLMFLUX_FORCE_REBUILD\" = \"1\" ] || [ ! -f \"$CONTAINERS_DIR/llm_processor.sif\" ]; then",
        "    echo \"Building Container in ${CONTAINERS_DIR}\"",
        "    export APPTAINER_DEBUG=1",
        "    apptainer build --force $CONTAINERS_DIR/llm_processor.sif $CONTAINER_DEF",
        "    echo Container built successfully: ${?}",
        "fi",
        "",
        "# Start server with clean environment",
        "# All APPTAINERENV_* variables are automatically passed in (prefix removed)",
        *([
            "# Find a consecutive free port on this compute node",
            "find_free_port() {",
            "    local port=$1",
            "    while ss -tlnH 2>/dev/null | awk '{print $4}' | grep -q \":${port}$\"; do",
            "        port=$((port + 1))",
            "    done",
            "    echo $port",
            "}",
            "OLLAMA_PORT=$(find_free_port ${OLLAMA_PORT:-8000})",
            "export APPTAINERENV_OLLAMA_PORT=$OLLAMA_PORT",
            "export APPTAINERENV_OLLAMA_HOST=\"0.0.0.0:$OLLAMA_PORT\"",
            "export APPTAINERENV_OLLAMA_API_KEY=\"$LLMFLUX_API_KEY\"",
            "echo \"Using port: $OLLAMA_PORT\"",
            "",
            "# Install cleanup trap early so the connection file (contains the API key)",
            "# and server are removed even if the job is cancelled with scancel (SIGTERM).",
            "CONNECTION_FILE=\"$HOME/.llmflux/serve/$SLURM_JOB_ID/connection.json\"",
            # Kill this job's server by PID, not only by pattern. `pkill -f
            # "ollama serve"` does match here — this path really does exec
            # `ollama serve` — but it matches the same user's OTHER jobs on the
            # node too, and $OLLAMA_PID was never killed at all. Same shape as
            # the vLLM trap, so the two engines cannot drift.
            "trap 'rm -f \"$CONNECTION_FILE\"; "
            "[ -n \"${OLLAMA_PID:-}\" ] && kill \"$OLLAMA_PID\" 2>/dev/null; "
            "pkill -f \"ollama serve\" || true' EXIT TERM INT",
            "",
        ] if mode == "serve" else []),
        "# Pass SLURM-allocated GPU(s) into the container (--cleanenv strips",
        "# CUDA_VISIBLE_DEVICES). Without this the container receives the list",
        "# synthesised from the *requested* GPU count rather than the devices",
        "# actually granted, and Ollama warns: \"user overrode visible devices\".",
        "export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}",
        "OLLAMA_DEBUG=1 apptainer exec --nv --cleanenv \\",
        "    --bind $DATA_INPUT_DIR:/app/data/input,$DATA_OUTPUT_DIR:/app/data/output,$MODELS_DIR:/app/models,$LOGS_DIR:/app/logs,$OLLAMA_HOME:$OLLAMA_HOME \\",
        "    $CONTAINERS_DIR/llm_processor.sif \\",
        "    ollama serve &",
        "",
        "OLLAMA_PID=$!",
        "",
        "# Wait for server",
        "for i in {1..60}; do",
        "    if curl -s \"http://localhost:$OLLAMA_PORT/api/version\" &>/dev/null; then",
        "        echo \"Ollama server started\"",
        "        break",
        "    fi",
        "    if ! ps -p $OLLAMA_PID > /dev/null; then",
        "        echo \"Ollama server died\"",
        "        exit 1",
        "    fi",
        "    echo \"Waiting... ($i/60)\"",
        "    sleep 1",
        "done",
        "",
        "# Pull model if needed",
        "MODEL_NAME=\"${OLLAMA_MODEL_NAME:-llama3.2:3b}\"",
        "echo \"Checking if model ${MODEL_NAME} exists...\"",
        "",
        # "# Extract base model name for Ollama (e.g. llama3.2:3b -> llama3.2)",
        # "if [[ \"$MODEL_NAME\" == *\":\"* ]]; then",
        # "    BASE_MODEL=$(echo \"$MODEL_NAME\" | cut -d':' -f1)",
        # "    echo \"Using base model name for Ollama: $BASE_MODEL\"",
        # "else",
        # "    BASE_MODEL=\"$MODEL_NAME\"",
        # "fi",
        "",
        "# Check if model exists, try to pull if it doesn't",
        "if ! curl -s \"http://localhost:$OLLAMA_PORT/api/tags\" | grep -q \"\\\"name\\\":\\\"$MODEL_NAME\\\"\"; then",
        "    echo \"Model not found, pulling base model ${MODEL_NAME}...\"",
        "    curl -X POST \"http://localhost:$OLLAMA_PORT/api/pull\" -d '{\"name\": \"'\"$MODEL_NAME\"'\"}' -H \"Content-Type: application/json\"",
        "    if [ $? -ne 0 ]; then",
        "        echo \"Failed to pull model ${MODEL_NAME}\"",
        "        echo \"Available models:\"",
        "        curl -s \"http://localhost:$OLLAMA_PORT/api/tags\" | grep -o '\"name\":\"[^\"]*\"' || echo \"None found\"",
        "        exit 1",
        "    else",
        "        echo \"Successfully pulled model ${MODEL_NAME}\"",
        "    fi",
        "else",
        "    echo \"Model ${MODEL_NAME} already exists\"",
        "fi",
        "",
        *([
            "# Write connection file for llmflux connect (restrict permissions — contains API key)",
            "# CONNECTION_FILE was defined earlier alongside the cleanup trap.",
            "(umask 077 && mkdir -p \"$(dirname $CONNECTION_FILE)\")",
            "chmod 700 \"$HOME/.llmflux\" \"$HOME/.llmflux/serve\" \"$(dirname $CONNECTION_FILE)\"",
            # Resolved once and reused everywhere this job publishes an address,
            # so the connection file and the notification email cannot disagree.
            # Previously all three sites called $(hostname) independently.
            #
            # hostname returns the FQDN on these compute nodes, and an FQDN here
            # resolves to IPv6 link-local ONLY — a client dialling it gets
            # nothing. The short name is resolvable.
            #
            # Unlike vLLM there is no nodes>1 branch: topology.resolve rejects
            # Ollama above one node at submit time, so the short name is the
            # only address this path can ever publish.
            "LLMFLUX_ENDPOINT_HOST=\"$(hostname -s)\"",
            "(umask 077 && cat > \"$CONNECTION_FILE\" <<EOF",
            "{",
            "  \"job_id\": \"$SLURM_JOB_ID\",",
            "  \"node\": \"$LLMFLUX_ENDPOINT_HOST\",",
            "  \"port\": $OLLAMA_PORT,",
            "  \"model\": \"$OLLAMA_MODEL_NAME\",",
            "  \"api_key\": \"$LLMFLUX_API_KEY\",",
            "  \"engine\": \"ollama\",",
            "  \"started_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"",
            "}",
            "EOF",
            ")",
            "chmod 600 \"$CONNECTION_FILE\"",
            "echo \"Connection info written to $CONNECTION_FILE\"",
            "",
            "# Send ready notification email with connection details",
            f"LLMFLUX_EMAIL={shlex.quote(email)}",
            "mail -s \"LLMFlux serve job $SLURM_JOB_ID is ready\" -- \"$LLMFLUX_EMAIL\" <<MAIL_EOF",
            "Your LLMFlux serve job has finished loading and is ready to use.",
            "",
            "  Job ID:   $SLURM_JOB_ID",
            "  Endpoint: http://$LLMFLUX_ENDPOINT_HOST:$OLLAMA_PORT/v1",
            "  API Key:  $LLMFLUX_API_KEY",
            "  Model:    $OLLAMA_MODEL_NAME",
            "  Engine:   ollama",
            "",
            "Example usage:",
            "",
            "  from openai import OpenAI",
            "  client = OpenAI(base_url=\"http://$LLMFLUX_ENDPOINT_HOST:$OLLAMA_PORT/v1\", api_key=\"$LLMFLUX_API_KEY\")",
            "  response = client.chat.completions.create(",
            "      model=\"$OLLAMA_MODEL_NAME\",",
            "      messages=[{\"role\": \"user\", \"content\": \"Hello!\"}]",
            "  )",
            "",
            "To get connection details again:",
            "  llmflux connect $SLURM_JOB_ID",
            "MAIL_EOF",
            "",
            "# Keep alive until wall time or scancel",
            "wait $OLLAMA_PID",
            # Without this the script's last command is the cleanup `if` block,
            # so a serve job whose server collapsed still exits 0 and sacct says
            # COMPLETED — `#SBATCH --mail-type=FAIL` never fires and the user is
            # told nothing. Captured immediately: anything between `wait` and
            # here overwrites $?.
            "LLMFLUX_SERVE_RC=$?",
            "# scancel and Ctrl-C are how a serve job normally ends; those are not",
            "# failures.",
            "case \"$LLMFLUX_SERVE_RC\" in 130|143) LLMFLUX_SERVE_RC=0 ;; esac",
        ] if mode == "serve" else [
            "# Run processor",
            f"{shlex.quote(sys.executable)} -c \"",
            "import os",
            "from llmflux.core.config import Config",
            "from llmflux.processors import BatchProcessor",
            "",
            "# Ensure OLLAMA environment variables are available in Python",
            "ollama_port = os.environ.get('OLLAMA_PORT')",
            "if ollama_port:",
            "    os.environ['OLLAMA_HOST'] = f'http://localhost:{ollama_port}'",
            "",
            "# Load model configuration",
            "config = Config()",
            "model_identifier = os.environ.get('APPTAINERENV_MODEL_IDENTIFIER', 'Llama-3.2-3B-Instruct')",
            "engine = os.environ.get('APPTAINERENV_ENGINE', 'ollama')",
            "",
            "try:",
            "    model_config = config.load_model_config(model_identifier)",
            "    model_config.engine = engine",
            "except Exception as e:",
            "    print(f'Error loading model config for {model_identifier}: {e}')",
            "    model_config = config.load_model_config('Llama-3.2-3B-Instruct')",
            "    model_config.engine = engine",
            "",
            "# Create batch processor with JSONL input",
            "batch_processor = BatchProcessor(",
            "    model_config=model_config,",
            "    batch_size=int(os.environ.get('BATCH_SIZE', '4')),",
            "    save_frequency=int(os.environ.get('SAVE_FREQUENCY', '50')),",
            "    max_retries=int(os.environ.get('MAX_RETRIES', '3')),",
            "    retry_delay=float(os.environ.get('RETRY_DELAY', '1.0'))",
            ")",
            "",
            "# Prepare run kwargs",
            "run_kwargs = {}",
            "",
            "# Add any other kwargs from environment variables",
            "for key in ['max_tokens', 'temperature', 'top_p', 'top_k']:",
            "    if key.upper() in os.environ:",
            "        run_kwargs[key] = os.environ[key.upper()]",
            "",
            f"batch_processor.run('{input_file}', '{output_file}', 'ollama', **run_kwargs)",
            "\"",
            "# Capture the processor's status before cleanup overwrites $?.",
            "BATCH_RC=$?",
        ]),
        "",
        "# Cleanup",
        # In serve mode the EXIT/TERM/INT trap handles connection-file removal
        # and killing the server, so no explicit cleanup is needed here.
        *( ["pkill -f \"ollama serve\" || true"] if mode != "serve" else [] ),
        "# Only remove temporary directories that we created",
        "if [ -d \"$APPTAINER_TMPDIR\" ] && [ -w \"$APPTAINER_TMPDIR\" ]; then",
        "    rm -rf \"$APPTAINER_TMPDIR\"",
        "fi",
        "if [ -d \"$APPTAINER_CACHEDIR\" ] && [ -w \"$APPTAINER_CACHEDIR\" ]; then",
        "    rm -rf \"$APPTAINER_CACHEDIR\"",
        "fi",
        *([
            "# Exit with the processor's status. Without this the script exits with\n"
            "# whatever cleanup returned, so a run whose every item failed still\n"
            "# reports success and the job looks complete.",
            "exit ${BATCH_RC:-0}",
        ] if mode != "serve" else [
            "# Exit with the server's status. Without this the script exits with\n"
            "# whatever cleanup returned, so a serve job whose server died still\n"
            "# reports success and the job looks complete.",
            "exit ${LLMFLUX_SERVE_RC:-0}",
        ]),
    ])
    return job_script
