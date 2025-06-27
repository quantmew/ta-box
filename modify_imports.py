#!/usr/bin/env python3
import os
import re
import glob

def modify_py_file(file_path):
    """修改.py文件中的导入语句"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找并替换导入语句
    # 匹配 from .ta_utility import 开头的行
    pattern = r'^(\s*)from \.ta_utility import (.+)$'
    
    def replace_import(match):
        indent = match.group(1)
        imports = match.group(2)
        return f'{indent}if not cython.compiled:\n{indent}    from .ta_utility import {imports}'
    
    new_content = re.sub(pattern, replace_import, content, flags=re.MULTILINE)
    
    # 如果内容有变化，写回文件
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"已修改: {file_path}")
        return True
    return False

def modify_pxd_file(file_path):
    """修改.pxd文件，在开头添加导入语句"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经包含导入语句
    if 'from .ta_utility cimport TA_INTEGER_DEFAULT' in content:
        return False
    
    # 在文件开头添加导入语句
    new_content = 'from .ta_utility cimport TA_INTEGER_DEFAULT\n' + content
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"已修改: {file_path}")
    return True

def main():
    # 获取所有.py文件
    py_files = glob.glob('tabox/ta_func/*.py')
    
    # 获取所有.pxd文件
    pxd_files = glob.glob('tabox/ta_func/*.pxd')
    
    modified_py = 0
    modified_pxd = 0
    
    # 修改.py文件
    for py_file in py_files:
        if modify_py_file(py_file):
            modified_py += 1
    
    # 修改.pxd文件
    for pxd_file in pxd_files:
        if modify_pxd_file(pxd_file):
            modified_pxd += 1
    
    print(f"\n修改完成:")
    print(f"修改了 {modified_py} 个.py文件")
    print(f"修改了 {modified_pxd} 个.pxd文件")

if __name__ == '__main__':
    main() 