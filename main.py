import importlib
import os
import sys
import inspect
from pathlib import Path

import yaml
import pynvml

from src.server.fedavg import FedAvgServer
from src.utils import utils

FLBENCH_ROOT = Path(__file__).parent.absolute()
if FLBENCH_ROOT not in sys.path:
    sys.path.append(FLBENCH_ROOT.as_posix())


from src.utils.tools import parse_args

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError(
            "No method is specified. Run like `python main.py <method> [config_file_relative_path] [cli_method_args ...]`,",
            "e.g., python main.py fedavg config/template.yml`",
        )
    
    
    method_name = sys.argv[1]

    config_file_path = None
    cli_method_args = []
    
    if len(sys.argv) > 2:        
        if ".yaml" in sys.argv[2] or ".yml" in sys.argv[2]:  # ***.yml or ***.yaml
            config_file_path = Path(sys.argv[2]).absolute()
            cli_method_args = sys.argv[3:]
        else:
            cli_method_args = sys.argv[2:]
    
    try:
        fl_method_server_module = importlib.import_module(f"src.server.{method_name}")
    except:
        raise ImportError(f"Can't import `src.server.{method_name}`.")
    
    module_attributes = inspect.getmembers(fl_method_server_module)
    server_class = [
        attribute
        for attribute in module_attributes
        if attribute[0].lower() == method_name + "server"
    ][0][1]

    get_method_hyperparams_func = getattr(server_class, f"get_hyperparams", None)

    config_file_args = None
    
    if config_file_path is not None and os.path.isfile(config_file_path):
        with open(config_file_path, "r") as f:
            try:
                config_file_args = yaml.safe_load(f)
                
            except:
                raise TypeError(
                    f"Config file's type should be yaml, now is {config_file_path}"
                )
    
    ARGS = parse_args(
        config_file_args, method_name, get_method_hyperparams_func, cli_method_args
    )
    
    utils.args = ARGS
    
    # target method is not inherited from FedAvgServer
    if server_class.__bases__[0] != FedAvgServer and server_class != FedAvgServer:
        parent_server_class = server_class.__bases__[0]
        if hasattr(parent_server_class, "get_hyperparams"):
            get_parent_method_hyperparams_func = getattr(
                parent_server_class, f"get_hyperparams", None
            )
            # class name: ***Server, only want ***
            parent_method_name = parent_server_class.__name__.lower()[:-6]
            # extract the hyperparams of parent method
            PARENT_ARGS = parse_args(
                config_file_args,
                parent_method_name,
                get_parent_method_hyperparams_func,
                cli_method_args,
            )
            setattr(ARGS, parent_method_name, getattr(PARENT_ARGS, parent_method_name))

    server = server_class(args=ARGS)
    server.run()
