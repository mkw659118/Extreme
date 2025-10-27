# coding : utf-8
# Author : yuxiang Zeng
import argparse
import importlib.util
import os
import sys
from dataclasses import fields
import glob
import shutil



def get_config(Config='MainConfig'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=Config)
    parser.add_argument('--config_path', type=str, default=f'configs/{Config}.py')
    args, unknown_args = parser.parse_known_args()

    # 动态设置 config_path
    args.config_path = f'configs/{args.exp_name}.py'
    args = load_config(args.config_path, args.exp_name)
    args = update_config_from_args(args, unknown_args)
    
    
    # 自动清理无效日志和空的 __pycache__ 文件夹
    clear_useless_logs()
    remove_pycache()
    return args


def load_config(file_path, class_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    print(module, class_name)
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




# 清理无效日志文件
def clear_useless_logs():
    for dirpath, _, _ in os.walk('./results/'):
        if 'log' in dirpath:
            for log_file in glob.glob(os.path.join(dirpath, '*.md')):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        if 'Round=1' not in f.read():
                            os.remove(log_file)
                except Exception as e:
                    print(f"Error processing file {log_file}: {e}")

# 删除空pycache文件夹
def remove_pycache(root_dir="."):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            shutil.rmtree(pycache_path)
    print("✅ All __pycache__ folders removed")
    return True

