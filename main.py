from __future__ import annotations

import cv2
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from ultralytics import YOLO
import os

class ObjectDetector:
    """物件偵測器"""
    def __init__(self, log_dir: str = 'logs'):
        self.yolo = YOLO('yolov8x.pt')
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 定義要檢測的物體類別
        self.TRACK_CLASSES = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            7: 'truck', 14: 'bird', 15: 'cat', 16: 'dog',
            24: 'backpack', 26: 'handbag', 28: 'suitcase'
        }
        
        # 為不同類別定義不同的顏色 (BGR格式)
        self.COLORS = {
            'person': (0, 0, 255),      # 紅色
            'bicycle': (255, 0, 0),     # 藍色
            'car': (0, 255, 0),         # 綠色
            'motorcycle': (255, 255, 0), # 青色
            'truck': (128, 0, 128),     # 紫色
            'bird': (255, 165, 0),      # 橙色
            'cat': (255, 192, 203),     # 粉色
            'dog': (255, 255, 0),       # 黃色
            'backpack': (165, 42, 42),  # 棕色
            'handbag': (0, 255, 255),   # 黃色
            'suitcase': (128, 128, 128) # 灰色
        }

    def analyze_and_visualize(self, image_path: str | Path, save_log: bool = True) -> Tuple[Optional[Dict], Optional[Any]]:
        """分析圖片並視覺化結果"""
        try:
            original_image = cv2.imread(str(image_path))
            if original_image is None:
                logging.error(f"無法讀取圖片: {image_path}")
                return None, None
                
            image_with_boxes = original_image.copy()
            
            # 初始化結果
            result = {
                'detections': {},  # 用於存儲每種物體的檢測結果
                'total_count': 0,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 執行檢測
            detections = self.yolo(str(image_path))
            
            # 處理檢測結果
            for det in detections[0].boxes.data:
                class_id = int(det[5])
                
                # 檢查是否是我們想要檢測的類別
                if class_id in self.TRACK_CLASSES:
                    class_name = self.TRACK_CLASSES[class_id]
                    confidence = float(det[4])
                    bbox = [int(x) for x in det[:4]]
                    
                    # 計算相對深度/距離
                    foot_y = bbox[3]
                    height = bbox[3] - bbox[1]
                    distance = (foot_y * height) / 10000
                    
                    # 將檢測結果添加到相應類別
                    if class_name not in result['detections']:
                        result['detections'][class_name] = []
                    
                    result['detections'][class_name].append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'distance': distance
                    })
                    
                    # 更新總數
                    result['total_count'] += 1
                    
                    # 在圖片上繪製標記
                    color = self.COLORS.get(class_name, (0, 255, 0))
                    cv2.rectangle(image_with_boxes, 
                                (bbox[0], bbox[1]), 
                                (bbox[2], bbox[3]), 
                                color, 2)
                    
                    # 添加標籤
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(image_with_boxes, label, 
                              (bbox[0], bbox[1]-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, color, 2)
                    
                    # 添加距離標籤
                    distance_label = f"Distance: {distance:.2f}m"
                    cv2.putText(image_with_boxes, distance_label,
                              (bbox[0], bbox[1]-30),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, color, 2)
            
            # 在圖片上添加總結信息
            summary = f"Total Objects: {result['total_count']}"
            cv2.putText(image_with_boxes, summary,
                      (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      1, (255, 255, 255), 2)
            
            # 添加時間戳
            timestamp = result['timestamp']
            cv2.putText(image_with_boxes, timestamp,
                      (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.7, (255, 255, 255), 2)
            
            # 保存標記後的圖片
            if save_log and result['total_count'] > 0:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                log_filename = self.log_dir / f"log-{timestamp}.jpg"
                cv2.imwrite(str(log_filename), image_with_boxes)
                result['log_file'] = str(log_filename)
                print(f"已保存檢測結果：{log_filename}")
            
            return result, image_with_boxes
            
        except Exception as e:
            print(f"分析圖片時發生錯誤：{str(e)}")
            return None, None

    def print_detection_results(self, result: Dict[str, Any]) -> None:
        """打印檢測結果"""
        print(f"\n時間：{result['timestamp']}")
        print(f"檢測到的物體總數：{result['total_count']}")
        
        if result['total_count'] > 0:
            print("\n各類別檢測結果：")
            for class_name, detections in result['detections'].items():
                print(f"\n{class_name}:")
                print(f"  數量: {len(detections)}")
                for i, det in enumerate(detections, 1):
                    print(f"  {i}. 距離: {det['distance']:.2f}m, "
                          f"置信度: {det['confidence']:.2f}")
        else:
            print("未檢測到任何物體")

def main() -> None:
    """主程式"""
    # 初始化設置
    os.system('python setup.py')

    print("系統開始運行...")
    
    try:
        while True:
            # 生成新圖片
            os.system('python image_composer.py')
            
            # 找最新的合成圖片
            images = list(Path('output').glob('*.[jp][pn][g]'))
            if images:
                latest_image = max(images, key=lambda x: x.stat().st_mtime)
                
                # 執行物件偵測
                detector = ObjectDetector(log_dir='logs')
                result, marked_image = detector.analyze_and_visualize(latest_image)
                
                if result is not None:
                    detector.print_detection_results(result)
                else:
                    print("圖片分析失敗")
            else:
                print("未找到可分析的圖片")
            
            # 等待下一次處理
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n程式已停止")
    except Exception as e:
        print(f"程式執行時發生錯誤：{str(e)}")

if __name__ == "__main__":
    main()