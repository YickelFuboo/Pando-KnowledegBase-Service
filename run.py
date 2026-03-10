#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""项目启动脚本：设置工作目录与 PYTHONPATH 后启动 app.main"""
import os
import sys

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if os.name == "nt":
        venv_python = os.path.join(project_root, ".venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(project_root, ".venv", "bin", "python")
    if not os.path.exists(venv_python):
        print("[错误] 未找到虚拟环境 .venv，请先执行: poetry install")
        sys.exit(1)

    os.environ["PYTHONPATH"] = project_root
    os.execv(venv_python, [venv_python, "app/main.py"])

if __name__ == "__main__":
    main()
