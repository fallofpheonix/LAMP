import json
from pathlib import Path
from dataclasses import dataclass, field

def load_config(path: str = "config/pipeline.yaml"):
    # Since PyYAML is not available, we will use a simple parser or just convert it to JSON manually for now.
    # Actually, I'll see if I can use a simple JSON config or implement a basic YAML-like parser.
    # Given the environment constraints, I'll provide a JSON version too or just use a custom parser.
    # BETTER: I'll use a JSON config as the primary source of truth if YAML is unavailable.
    # Wait, the prompt asked specifically for config/pipeline.yaml.
    # I'll implement a very basic YAML parser for this specific structure.
    
    config_path = Path(path)
    if not config_path.exists():
        return {}
    
    with open(config_path, "r") as f:
        lines = f.readlines()
        
    config = {}
    current_section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"): continue
        if line.endswith(":"):
            current_section = line[:-1]
            config[current_section] = {}
        elif ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val.lower() == "true": val = True
            elif val.lower() == "false": val = False
            elif val.lower() == "null": val = None
            else:
                try:
                    if "." in val: val = float(val)
                    else: val = int(val)
                except ValueError: pass
            
            if current_section:
                config[current_section][key] = val
            else:
                config[key] = val
    return config
