.. _llmflux:

LLMFlux
---------------

`LLMFlux <https://github.com/Center-for-AI-Innovation/llmflux>`_ is a module that allows users to run a Large Language Model (LLM) as a batch processing job with SLURM, using JSONL formatted files as input and allowing for customized model selection using either OLLAMA or vLLM as backend servers. LLMFlux uses an OpenAI-compatible API format to maximize it's utility.

LLMFlux is launched from the head node and handles all batch submission internally. It uses minimal head node resources to launch job. Once the job is submitted and the job is complete, LLMFlux writes the output to JSON for machine or human parsing.

- :ref:`llmflux-setup`
- :ref:`running-llmflux`

.. _llmflux-setup:
How to set up LLMFlux
~~~~~~~~~~~~~~~~~~

To set up LLMFlux, you will need to load the module.

.. code-block::

    module load llmflux

The module load will provide a conda environment you can activate that contains LLMFlux. To activate the environment:

.. code-block::

    conda activate base

With the environment active, you can run the following command to test that LLMFlux is active:

.. code-block::

    llmflux -h

A usage message should appear, indicating that LLMFlux is now ready to use.

.. _running-llmflux:
How to run LLMFlux
~~~~~~~~~~~~~~~~~~
LLMFlux includes a command-line interface for submitting batch processing jobs. To process a JSONL directly using the llama3.2 model with 3 billion parameters, you would find the corresponding name in `src/llmflux/templates/models.yaml` and search for `llama`:

.. code-block::

    ...
  Llama-3.2-3B-Instruct:
    name: llama3.2:3b
    hf_name: meta-llama/Llama-3.2-3B-Instruct
    resources:
      gpu_layers: 32
      gpu_memory: 16GB
      batch_size: 8
      max_concurrent: 2
    parameters:
      temperature: 0.7
      top_p: 0.9
      max_tokens: 2048
      stop_sequences:
      - '###'
    system:
      prompt: You are a helpful AI assistant. You are direct, accurate, and helpful in your responses.
    validation:
      temperature_range:
      - 0.0
      - 1.0
      max_tokens_limit: 4096
      batch_size_range:
      - 1
      - 16
      concurrent_range:
      - 1
      - 4
    requirements:
      min_gpu_memory: 16GB
      recommended_gpu: A100
      cuda_version: '>=12.0'
      cpu_threads: 4
      gpu_memory_utilization: 0.9

Every model block in this file starts with the model keyword that will be used by LLMFlux. This name will work for both ollama and vllm, within LLMFlux. If you are using vLLM as your engine, the model it will pull is listed as hf_name. If you are using Ollama, the name parameter gives the Ollama version of the name. If you have a custom model, you will need to add it to the yaml file based on an existing example.

To test your model, you can run a benchmark command.

Once you have selected which model, you will need to prepare the inputs JSONL file.

.. code-block::

    {"custom_id":"request1","method":"POST","url":"/v1/chat/completions","body":{"messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Explain quantum computing"}],"temperature":0.7,"max_tokens":500}}
    {"custom_id":"request2","method":"POST","url":"/v1/chat/completions","body":{"messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is machine learning?"}],"temperature":0.7,"max_tokens":500}}

Ensure that the JSONL has a model parameter under body that  matches the name in the configuration yaml file.

With the inputs prepared, you are ready to run LLMFlux. The command to run the basic JSONL above looks something like:

.. code-block::

    llmflux run --model Llama-3.2-3B-Instruct --input data/prompts.jsonl --output results/output.json

To run this same command, but using the vLLM backend:

.. code-block::

    llmflux run --model Llama-3.2-3B-Instruct --input data/prompts.jsonl --output results/output.json --engine=vllm

Note that at the moment the only two valid choices for engine are "vllm" or "ollama." In the future, we may add support for different engines.


