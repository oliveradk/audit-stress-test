# Intern Server Setup

The core eval server code does not need additional setup beyond what's specified in the README.md, 

the following is only required for running the internalization server with vllm through `/experiments/run_ssc_vllm.sh`

On a gpu-enabled machine:
```bash
# Create venv
uv venv

# Activate
source .venv/bin/activate

# Install bitsandbytes
uv pip install bitsandbytes

# Install vllm with automatic torch backend detection
uv pip install vllm --torch-backend=auto

# Install other dependencies
uv pip install -r requirements.txt
```
