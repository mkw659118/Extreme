# coding : utf-8
# Author : yuxiang Zeng
import argparse
import importlib.util
import sys
from dataclasses import fields


def get_config_by_name(config_name):
    # 配置类文件都放在 configs/ 目录下
    config_name += '_config'
    module_path = f"configs.{config_name}"
    module = importlib.import_module(module_path)
    config_class = getattr(module, config_name)
    return config_class()


def get_config(Config='MainConfig'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=f'configs/{Config}.py')
    parser.add_argument('--exp_name', type=str, default=Config)
    args, unknown_args = parser.parse_known_args()
    args = load_config(args.config_path, args.exp_name)
    args = update_config_from_args(args, unknown_args)
    return args


def load_config(file_path, class_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    config = getattr(module, class_name)()
    return config


def update_config_from_args(config, args):
    it = iter(args)
    for arg in it:
        if arg.startswith("--"):
            if "=" in arg:
                key, value = arg[2:].split("=")
            else:
                key = arg[2:]
                value = next(it)

            field_type = next((f.type for f in fields(config) if f.name == key), str)
            if field_type == bool:
                value = value.lower() in ['true', '1', 'yes']
            else:
                value = field_type(value)
            setattr(config, key, value)
    return config