from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import os
import time
import glob

class TrackObjectDetector:
    def __init__(self, log_dir='logs', max_logs=10):
        self.yolo = YOLO('yolov8x.pt')
        self.log_dir = log_dir
        self.max_logs = max_logs
        
        # 定義要檢測的物體類別
        self.TRACK_CLASSES = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            7: 'truck',
            14: 'bird',
            15: 'cat',
            16: 'dog',
            24: 'backpack',
            26: 'handbag',
            28: 'suitcase'
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
        
        # 創建日誌目錄
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    def manage_log_files(self):
        log_files = glob.glob(os.path.join(self.log_dir, 'log-*.jpg'))
        log_files.sort(key=lambda x: os.path.getmtime(x))
        
        if len(log_files) > self.max_logs:
            files_to_delete = len(log_files) - self.max_logs
            oldest_files = log_files[:files_to_delete]
            
            for file in oldest_files:
                try:
                    os.remove(file)
                    print(f"已刪除舊日誌文件：{file}")
                except Exception as e:
                    print(f"刪除文件 {file} 時發生錯誤：{e}")
        
    def analyze_and_visualize(self, image_path, save_log=True):
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Error: Unable to read image from {image_path}")
            return None, None
            
        image_with_boxes = original_image.copy()
        
        # 初始化結果
        result = {
            'detections': {},  # 用於存儲每種物體的檢測結果
            'total_count': 0
        }
        
        # 執行檢測
        detections = self.yolo(image_path)
        
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
        
        # 保存標記後的圖片
        if save_log and result['total_count'] > 0:  # 只在檢測到物體時保存
            self.manage_log_files()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            log_filename = os.path.join(self.log_dir, f"log-{timestamp}.jpg")
            cv2.imwrite(log_filename, image_with_boxes)
            result['log_file'] = log_filename
        
        return result, image_with_boxes

def main():
    detector = TrackObjectDetector(log_dir='logs', max_logs=10)
    image_path = 'images/image.png'  # 請修改為您的圖片路徑
    
    try:
        while True:
            result, marked_image = detector.analyze_and_visualize(image_path)
            
            if result is not None:
                print(f"\n時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"檢測到的物體總數：{result['total_count']}")
                
                if result['total_count'] > 0:
                    print("\n各類別檢測結果：")
                    for class_name, detections in result['detections'].items():
                        print(f"\n{class_name}:")
                        print(f"  數量: {len(detections)}")
                        for i, det in enumerate(detections, 1):
                            print(f"  {i}. 距離: {det['distance']:.2f}m, "
                                  f"置信度: {det['confidence']:.2f}")
                    
                    if 'log_file' in result:
                        print(f"\n已保存標記圖片：{result['log_file']}")
                else:
                    print("未檢測到任何物體")
            
            time.sleep(1)  # 可調整檢測間隔
            
    except KeyboardInterrupt:
        print("\n程式已停止")
    except Exception as e:
        print(f"發生錯誤：{e}")

if __name__ == "__main__":
    main()