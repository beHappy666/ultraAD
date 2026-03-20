import os
import subprocess
import json

class UltraADAdapter:
    """
    Adapter for interacting with the ultraAD codebase.
    """

    def __init__(self, ultraad_root=None):
        self.root = ultraad_root or self._find_root()
        if not self.root:
            raise FileNotFoundError("Could not find the ultraAD root directory.")

    def _find_root(self):
        """
        Find the root of the ultraAD project by looking for a known file.
        """
        current_dir = os.getcwd()
        while current_dir != os.path.dirname(current_dir):
            if "CLAUDE.md" in os.listdir(current_dir) and "scripts" in os.listdir(current_dir):
                return current_dir
            current_dir = os.path.dirname(current_dir)
        return None

    def get_config(self, config_name="VAD_tiny_debug.py"):
        """
        Read a configuration file from the ultraAD configs directory.
        """
        config_path = os.path.join(self.root, "configs", config_name)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            return f.read()

    def run_debug_one_sample(self, config_name="VAD_tiny_debug.py"):
        """
        Execute the debug_one_sample.py script from ultraAD.
        """
        script_path = os.path.join(self.root, "scripts", "debug_one_sample.py")
        config_path = os.path.join(self.root, "configs", config_name)

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        result = subprocess.run(
            ["python", script_path, config_path],
            capture_output=True,
            text=True,
            cwd=self.root,
        )
        return result

    def add_plugin_config(self, plugin_name: str, plugin_config: dict):
        """
        A placeholder method to demonstrate managing plugin configurations.
        This might involve modifying a JSON/YAML file that lists active plugins.
        """
        # For demonstration, we'll just print the action.
        # In a real scenario, this would modify a file.
        print(f"Adding config for plugin '{plugin_name}' to ultraAD.")

        # Example: update a features.json file
        features_path = os.path.join(self.root, "configs", "features.json")
        features = {}
        if os.path.exists(features_path):
            with open(features_path, "r") as f:
                features = json.load(f)

        features[plugin_name] = plugin_config

        with open(features_path, "w") as f:
            json.dump(features, f, indent=2)

        return features_path
