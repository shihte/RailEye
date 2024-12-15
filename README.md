# RailEye (鐵道之眼) - 智能輕軌自動駕駛系統

## 專案簡介
RailEye (鐵道之眼) 是一套結合深度學習與自動控制的輕軌系統解決方案。透過 YOLOv8 卷積神經網路技術，系統能夠即時監控與分析站台環境，不僅提供精確的物體識別，更整合自動駕駛功能，實現智能化的列車停靠與運行控制。本系統致力於提升輕軌運輸的安全性、效率性與可靠性。

## 核心功能
- 🚊 自動駕駛系統
  - 智能速度控制
  - 精準停站定位
  - 自動緊急制動
  - 平穩加減速控制
  - 車距自動調節

- 🎯 即時物體偵測
  - 人員動態追蹤
  - 交通工具識別
  - 動物行為監測
  - 物品遺留偵測

- 🔍 智能分析功能
  - 精確距離估算
  - 軌道環境分析
  - 站台擁擠度評估
  - 異常事件預警

- 📊 自動記錄系統
  - 運行數據記錄
  - 事件影像保存
  - 自動駕駛日誌
  - 系統狀態監控

## 技術特點
- 深度學習技術
  - YOLOv8x 即時物體偵測
  - 自定義訓練資料集
  - 高效能運算優化

- 自動控制系統
  - PID 控制器調校
  - 模糊邏輯控制
  - 自適應巡航系統
  - 精確制動控制

- 系統整合
  - 多感測器融合
  - 分散式架構
  - 即時通訊協定
  - 安全備援機制

## 應用場景
- 輕軌自動駕駛
- 智能站台管理
- 安全監控系統
- 運行效能優化
- 事故預防預警

## 系統架構
### 硬體需求
- 工業級電腦
- CUDA 支援的 NVIDIA GPU
- 高解析度攝影機
- 車載感測器套件

### 軟體需求
- Python 3.8+
- 深度學習框架
  - ultralytics
  - opencv-python
  - numpy
  - pandas
- 自動控制模組
  - control
  - scipy
  - pyserial

## 安裝配置
1. 系統安裝
```bash
git clone https://github.com/yourusername/RailEye.git
cd RailEye
```

2. 環境設定
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 套件安裝
```bash
pip install -r requirements.txt
```

## 使用說明
1. 系統初始化
```bash
python setup.py
```

2. 啟動系統
```bash
python main.py
```

3. 參數配置
```python
# config.py
SETTINGS = {
    # 自動駕駛參數
    'auto_drive': {
        'max_speed': 40,        # 最高速度（km/h）
        'brake_distance': 100,  # 制動距離（m）
        'acceleration': 0.8     # 加速度（m/s²）
    },
    # 物體偵測參數
    'detection': {
        'max_logs': 10,        # 日誌保存數量
        'interval': 0.1,       # 檢測間隔（秒）
        'confidence': 0.5      # 置信度閾值
    },
    # 系統配置
    'system': {
        'debug_mode': False,   # 除錯模式
        'save_images': True,   # 保存偵測圖片
        'auto_backup': True    # 自動備份
    }
}
```

## 系統輸出
- 即時監控資訊
  - 列車運行狀態
  - 物體偵測結果
  - 系統運行參數
- 自動記錄
  - 運行日誌
  - 事件影像
  - 系統報告

## 安全機制
- 緊急制動系統
- 異常事件處理
- 系統備援切換
- 故障自動回報

## 未來規劃
- [ ] 多列車協同控制
- [ ] AI 預測性維護
- [ ] 5G 即時通訊整合
- [ ] 智能調度最佳化
- [ ] 新一代感測器整合

## 授權資訊
本專案採用 MIT 授權條款 - 詳細內容請參閱 [LICENSE](LICENSE) 檔案。

## 技術支援
- 文件：[文件連結]
- 問題回報：[Issue 連結]
- 技術社群：[社群連結]