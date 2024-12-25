from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

class Setup:
    """設定和管理系統資源"""
    def __init__(self, base_dir: str = '.'):
        self.base_path = Path(base_dir)
        self.directories = {
            'images': self.base_path / 'images',
            'output': self.base_path / 'output',
            'logs': self.base_path / 'logs'
        }
        self.source_directories = {
            'backgrounds': self.directories['images'] / 'backgrounds',
            'cats': self.directories['images'] / 'cats',
            'dogs': self.directories['images'] / 'dogs',
            'peoples': self.directories['images'] / 'peoples'
        }
        self.log_file = self.base_path / 'system.log'

    def setup_directories(self) -> bool:
        """建立必要的目錄結構"""
        try:
            # 建立主要目錄
            for directory in self.directories.values():
                directory.mkdir(parents=True, exist_ok=True)
            
            # 建立來源圖片目錄
            for source_dir in self.source_directories.values():
                source_dir.mkdir(parents=True, exist_ok=True)
            
            return True
        except Exception as e:
            print(f"建立目錄時發生錯誤: {e}")
            return False

    def setup_logging(self) -> None:
        """設置日誌系統"""
        try:
            logging.basicConfig(
                filename=self.log_file,
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                encoding='utf-8'
            )
            logging.info("日誌系統初始化成功")
        except Exception as e:
            print(f"設置日誌系統時發生錯誤: {e}")

    def verify_model(self, model_name: str = 'yolov8x.pt') -> bool:
        """驗證 YOLO 模型是否存在"""
        model_path = self.base_path / model_name
        exists = model_path.exists()
        if not exists:
            print(f"找不到 YOLO 模型: {model_path}")
            logging.error(f"找不到 YOLO 模型: {model_path}")
        return exists

    def verify_source_images(self) -> bool:
        """驗證源圖片目錄是否包含圖片"""
        for category, path in self.source_directories.items():
            images = list(path.glob('*.[jp][pn][g]'))
            if not images:
                print(f"警告: {category} 目錄中沒有圖片")
                logging.warning(f"{category} 目錄中沒有圖片")
                return False
        return True

    def cleanup_old_files(self, directory: str, keep: int = 1) -> None:
        """清理舊檔案，保留最新的幾個"""
        dir_path = self.directories.get(directory)
        if not dir_path or not dir_path.exists():
            return

        # 特別處理不同目錄的保留數量
        if directory == 'output':
            keep = 1  # output 目錄只保留1張
        elif directory == 'logs':
            keep = 15  # logs 目錄保留15個檔案
        
        files = list(dir_path.glob('*.[jp][pn][g]'))
        if directory == 'logs':
            files = list(dir_path.glob('*.log'))  # 對於 logs 目錄，搜尋 .log 檔案
        
        if len(files) > keep:
            # 按修改時間排序
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            # 刪除舊檔案
            for file in files[keep:]:
                try:
                    file.unlink()
                    logging.info(f"已刪除舊檔案: {file}")
                except Exception as e:
                    logging.error(f"刪除檔案失敗 {file}: {e}")

    def get_latest_image(self, directory: str) -> Optional[Path]:
        """獲取指定目錄中最新的圖片"""
        dir_path = self.directories.get(directory)
        if not dir_path or not dir_path.exists():
            return None

        files = list(dir_path.glob('*.[jp][pn][g]'))
        if not files:
            return None

        return max(files, key=lambda x: x.stat().st_mtime)

    def initialize(self) -> bool:
        """執行完整的初始化流程"""
        try:
            # 設置日誌
            self.setup_logging()
            logging.info("開始系統初始化")

            # 建立目錄
            if not self.setup_directories():
                return False

            # 驗證模型
            if not self.verify_model():
                return False

            # 驗證源圖片
            if not self.verify_source_images():
                logging.warning("源圖片驗證失敗，但系統將繼續執行")

            # 清理舊檔案
            self.cleanup_old_files('output', keep=1)
            self.cleanup_old_files('logs', keep=15)

            logging.info("系統初始化完成")
            return True

        except Exception as e:
            error_msg = f"初始化過程發生錯誤: {e}"
            print(error_msg)
            logging.error(error_msg)
            return False

def main() -> None:
    """測試設置功能"""
    setup = Setup()
    if setup.initialize():
        print("系統初始化成功")
        print("\n目錄結構:")
        print("- images/")
        print("  ├── backgrounds/")
        print("  ├── cats/")
        print("  ├── dogs/")
        print("  └── peoples/")
        print("- output/")
        print("- logs/")
        print("\n使用說明:")
        print("1. 在 images 目錄的各子目錄中放入對應的圖片")
        print("2. 確保 yolov8x.pt 模型檔案存在")
        print("3. 執行 main.py 開始處理")
    else:
        print("系統初始化失敗，請檢查 system.log")

if __name__ == "__main__":
    main()