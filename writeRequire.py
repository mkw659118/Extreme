import subprocess

def generate_requirements_file(file_path):
    """运行 pip freeze 并将输出保存到指定文件，确保文件编码为 UTF-8"""
    try:
        # 运行 pip freeze 命令
        result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            print("Failed to run pip freeze")
            print(result.stderr)
            return

        # 将输出写入文件，确保使用 UTF-8 编码
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
        print(f"Successfully generated {file_path} with UTF-8 encoding")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    requirements_file = "requirements.txt"  # 指定生成的文件路径
    generate_requirements_file(requirements_file)