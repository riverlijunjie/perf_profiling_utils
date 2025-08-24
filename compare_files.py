#!/usr/bin/env python3
"""
文件比较工具 - 比较两个目录下对应文件名的文本文件
跳过每个文件的第一行，输出详细的比较结果
"""

import os
import sys
import argparse
import difflib
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class FileComparator:
    def __init__(self, dir1: str, dir2: str, skip_lines: int = 1):
        """
        初始化文件比较器
        
        Args:
            dir1: 第一个目录路径
            dir2: 第二个目录路径
            skip_lines: 跳过的行数，默认为1
        """
        self.dir1 = Path(dir1)
        self.dir2 = Path(dir2)
        self.skip_lines = skip_lines
        self.results = []
        
    def read_file_content(self, file_path: Path) -> Optional[List[str]]:
        """
        读取文件内容，跳过指定行数
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容行列表，如果读取失败返回None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 跳过指定行数
                return lines[self.skip_lines:] if len(lines) > self.skip_lines else []
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    lines = f.readlines()
                    return lines[self.skip_lines:] if len(lines) > self.skip_lines else []
            except:
                print(f"warning: cannot read {file_path}")
                return None
        except Exception as e:
            print(f"error: failed to read file {file_path}: {e}")
            return None
    
    def get_file_list(self, directory: Path, pattern: str = "*") -> List[Path]:
        """
        获取目录下的文件列表
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            
        Returns:
            文件路径列表
        """
        if not directory.exists():
            print(f"error: directory {directory} doesn't exist")
            return []
            
        files = []
        if directory.is_dir():
            files = list(directory.glob(pattern))
            # 只保留文件，排除目录
            files = [f for f in files if f.is_file()]
        
        return sorted(files)
    
    def compare_files(self, file1: Path, file2: Path) -> Dict:
        """
        比较两个文件
        
        Args:
            file1: 第一个文件路径
            file2: 第二个文件路径
            
        Returns:
            比较结果字典
        """
        result = {
            'file1': str(file1),
            'file2': str(file2),
            'status': 'unknown',
            'differences': [],
            'stats': {}
        }
        
        # 检查文件是否存在
        if not file1.exists():
            result['status'] = 'file1_missing'
            return result
        if not file2.exists():
            result['status'] = 'file2_missing'
            return result
        
        # 读取文件内容
        content1 = self.read_file_content(file1)
        content2 = self.read_file_content(file2)
        
        if content1 is None or content2 is None:
            result['status'] = 'read_error'
            return result
        
        # 统计信息
        result['stats'] = {
            'file1_lines': len(content1),
            'file2_lines': len(content2),
            'file1_chars': sum(len(line) for line in content1),
            'file2_chars': sum(len(line) for line in content2)
        }
        
        # 比较内容
        if content1 == content2:
            result['status'] = 'identical'
        else:
            result['status'] = 'different'
            # 生成详细差异
            differ = difflib.unified_diff(
                content1, content2,
                fromfile=f"{file1.name} (跳过{self.skip_lines}行)",
                tofile=f"{file2.name} (跳过{self.skip_lines}行)",
                lineterm=''
            )
            result['differences'] = list(differ)
        
        return result
    
    def compare_directories(self, file_pattern: str = "*.txt") -> None:
        """
        比较两个目录下的所有对应文件
        
        Args:
            file_pattern: 文件匹配模式
        """
        print(f"start compare files:")
        print(f"directory 1: {self.dir1}")
        print(f"directory 2: {self.dir2}")
        print(f"file pattern: {file_pattern}")
        print(f"skip lines: {self.skip_lines}")
        print("-" * 60)
        
        # 获取两个目录的文件列表
        files1 = self.get_file_list(self.dir1, file_pattern)
        files2 = self.get_file_list(self.dir2, file_pattern)
        
        # 获取文件名集合
        names1 = {f.name for f in files1}
        names2 = {f.name for f in files2}
        
        # 找出所有需要比较的文件
        all_names = names1.union(names2)
        
        identical_count = 0
        different_count = 0
        missing_count = 0
        error_count = 0
        
        for name in sorted(all_names):
            file1 = self.dir1 / name
            file2 = self.dir2 / name
            
            result = self.compare_files(file1, file2)
            self.results.append(result)
            
            # 输出结果
            if result['status'] == 'identical':
                print(f"✓ {name}: file content is identical")
                identical_count += 1
            elif result['status'] == 'different':
                print(f"✗ {name}: file content is different")
                print(f"  file 1: {result['stats']['file1_lines']} lines, {result['stats']['file1_chars']} chars")
                print(f"  file 2: {result['stats']['file2_lines']} lines, {result['stats']['file2_chars']} chars")
                different_count += 1
            elif result['status'] == 'file1_missing':
                print(f"⚠ {name}: file only exists in directory 2")
                missing_count += 1
            elif result['status'] == 'file2_missing':
                print(f"⚠ {name}: file only exists in directory 1")
                missing_count += 1
            else:
                print(f"✗ {name}: read error")
                error_count += 1
        
        # 输出总结
        print("-" * 60)
        print("summary:")
        print(f"identical files: {identical_count}")
        print(f"different files: {different_count}")
        print(f"missing files: {missing_count}")
        print(f"error files: {error_count}")
        print(f"total files: {len(all_names)}")
    
    def save_detailed_report(self, output_file: str) -> None:
        """
        保存详细的比较报告
        
        Args:
            output_file: 输出文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("file comparison detailed report\n")
                f.write("=" * 60 + "\n\n")
                
                for result in self.results:
                    f.write(f"file: {Path(result['file1']).name}\n")
                    f.write(f"status: {result['status']}\n")
                    
                    if 'stats' in result and result['stats']:
                        f.write(f"statistics:\n")
                        f.write(f"  file 1: {result['stats'].get('file1_lines', 'N/A')} lines, "
                               f"{result['stats'].get('file1_chars', 'N/A')} chars\n")
                        f.write(f"  file 2: {result['stats'].get('file2_lines', 'N/A')} lines, "
                               f"{result['stats'].get('file2_chars', 'N/A')} chars\n")

                    if result['differences']:
                        f.write("differences:\n")
                        for line in result['differences']:
                            f.write(f"  {line}\n")
                    
                    f.write("-" * 40 + "\n\n")
            
            print(f"save report details to: {output_file}")
            
        except Exception as e:
            print(f"failed to save: {e}")


def main():
    parser = argparse.ArgumentParser(description="compare text files with the same name in two directories")
    parser.add_argument("dir1", help="path to the first directory")
    parser.add_argument("dir2", help="path to the second directory")
    parser.add_argument("-p", "--pattern", default="*.txt", help="file matching pattern (default: *.txt)")
    parser.add_argument("-s", "--skip", type=int, default=1, help="number of lines to skip (default: 1)")
    parser.add_argument("-o", "--output", help="path to the detailed report output file")
    parser.add_argument("--all-files", action="store_true", help="compare all file types")

    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.dir1):
        print(f"error: directory {args.dir1} doesn't exist")
        sys.exit(1)
    if not os.path.exists(args.dir2):
        print(f"error: directory {args.dir2} doesn't exist")
        sys.exit(1)
    
    # 设置文件模式
    pattern = "*" if args.all_files else args.pattern
    
    # 创建比较器并执行比较
    comparator = FileComparator(args.dir1, args.dir2, args.skip)
    comparator.compare_directories(pattern)
    
    # 保存详细报告
    if args.output:
        comparator.save_detailed_report(args.output)


if __name__ == "__main__":
    main()
