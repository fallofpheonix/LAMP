from pathlib import Path


def load_config(path: str = "config/pipeline.yaml"):
    config_path = Path(path)
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    config = {}
    current_section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":"):
            current_section = line[:-1]
            config[current_section] = {}
        elif ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            elif val.lower() == "null":
                val = None
            else:
                try:
                    if "." in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    pass

            if current_section:
                config[current_section][key] = val
            else:
                config[key] = val
    return config
