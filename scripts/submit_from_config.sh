#!/bin/bash
set -e

CONFIG_FILE="config.yaml"
TEMPLATE_FILE="job_template.slurm"
JOB_FILE="job_generated.slurm"

if [ ! -f "$CONFIG_FILE" ] || [ ! -f "$TEMPLATE_FILE" ]; then
  echo "Missing config or template"
  exit 1
fi

# Function to extract config values using Python/YAML
read_config() {
  python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE')).get('$1', ''))"
}

# Mapping of placeholders to config keys
declare -A config_map=(
  ["JOB_NAME"]="job_name"
  ["OUTPUT_FILE"]="output_file"
  ["ERROR_FILE"]="error_file"
  ["TIME_LIMIT"]="time_limit"
  ["GPUS"]="gpus"
  ["PARTITION"]="partition"
  ["NODELIST"]="nodelist"
  ["CPUS_PER_TASK"]="cpus_per_task"
  ["MEMORY"]="memory"
  ["CODE_DIR"]="code_dir"
  ["PROJECT_ROOT"]="project_root"
)

# Copy template to target job file
cp "$TEMPLATE_FILE" "$JOB_FILE"

# Substitute placeholders with config values
for placeholder in "${!config_map[@]}"; do
  value=$(read_config "${config_map[$placeholder]}")
  if [ -z "$value" ]; then
    echo "Warning: Config key '${config_map[$placeholder]}' is empty or missing"
  fi

  # Escape characters that might break sed (like slashes or ampersands)
  value_escaped=$(printf '%s\n' "$value" | sed 's/[&/\]/\\&/g')

  # Detect OS and use correct sed syntax
  if [[ "$OSTYPE" == "darwin"* ]]; then
      sed -i '' "s|{{${placeholder}}}|${value}|g" "$JOB_FILE"
  else
      sed -i "s|{{${placeholder}}}|${value}|g" "$JOB_FILE"
  fi
done

# Submit the job
chmod +x "$JOB_FILE"
sbatch "$JOB_FILE"
