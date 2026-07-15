"""Configure LIBERO to use assets from this repo checkout."""
import os
import sys


def ensure_libero_paths():
    import yaml

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_root = os.path.join(repo_root, "LIBERO", "libero", "libero")
    libero_config_dir = os.path.join(repo_root, "LIBERO", ".libero")
    os.makedirs(libero_config_dir, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = libero_config_dir

    config = {
        "benchmark_root": package_root,
        "bddl_files": os.path.join(package_root, "bddl_files"),
        "init_states": os.path.join(package_root, "init_files"),
        "datasets": os.path.join(repo_root, "LIBERO", "libero", "datasets"),
        "assets": os.path.join(package_root, "assets"),
    }
    config_file = os.path.join(libero_config_dir, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    libero_mod = sys.modules.get("libero.libero")
    if libero_mod is not None:
        libero_mod.libero_config_path = libero_config_dir
        libero_mod.config_file = config_file
