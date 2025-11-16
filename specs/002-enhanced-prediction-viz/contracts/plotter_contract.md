# Contract: plotter.py (視覺化繪圖模組)

**Module**: `src/visualization/plotter.py`
**Version**: 1.0.0
**Status**: Stable

## 函式: plot_dual_view

### 簽名

```python
def plot_dual_view(
    prediction_result: Dict[str, Any],
    output_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    繪製雙視圖 (3分類 + 5分類) 橫條圖並儲存為PNG檔案

    Args:
        prediction_result: PredictionResultDict (data-model.md定義)
        output_path: 輸出檔案路徑,若為 None 則自動生成
        config: VisualizationConfigDict (data-model.md定義),若為 None 使用預設配置

    Returns:
        str: 實際儲存的檔案路徑

    Raises:
        ValueError: 若 prediction_result 格式不正確
        IOError: 若檔案寫入失敗

    Example:
        >>> result = {...}  # PredictionResultDict
        >>> path = plot_dual_view(result, "output/prediction.png")
        >>> print(f"圖表已儲存至: {path}")
    """
```

### 前置條件

1. `prediction_result` 必須包含 PredictionResultDict 所有必要欄位
2. 若指定 `output_path`,其父目錄必須存在或可建立
3. 系統必須安裝 matplotlib 3.8+

### 後置條件

1. PNG 檔案成功儲存至指定路徑
2. 圖表包含兩個子圖: 上方為3分類,下方為5分類
3. 所有橫條從 x=0 開始 (左側對齊)
4. 中文字型正確顯示
5. 若實際資料存在,標示預測正確性 (✓或✗符號)
6. 圖片解析度滿足 spec SC-005 (清晰可讀於1920x1080)

### 視覺化規範

#### 3分類子圖 (上方)
- 標題: "3分類聚合預測"
- Y軸: 三個類別 (由上至下: 看跌, 震盪, 看漲)
- X軸: 機率 (0.0 ~ 1.0)
- 顏色: 使用 config.colors_3class
- 紋理: 使用 config.hatches_3class (色盲輔助)
- 標籤: 每個橫條右側顯示百分比 (如 "35.0%")
- 標記: 預測類別左側加上 "←" 箭頭

#### 5分類子圖 (下方)
- 標題: "5分類詳細預測"
- Y軸: 五個類別 (由上至下: 極度下跌, 溫和下跌, 區間震盪, 溫和上漲, 極度上漲)
- X軸: 機率 (0.0 ~ 1.0)
- 顏色: 使用 config.colors_5class
- 標籤: 每個橫條右側顯示百分比
- 標記: 預測類別左側加上 "←" 箭頭

#### 實際資料標記 (若存在)
- 在標題下方顯示實際資料摘要:
  - "實際收盤價: {價格} ({漲跌幅})"
  - "5分類預測: {'正確✓' if is_correct_5class else '錯誤✗'}"
  - "3分類預測: {'正確✓' if is_correct_3class else '錯誤✗'}"
- 使用綠色顯示正確,紅色顯示錯誤

### 錯誤處理

| 錯誤情境 | 例外型別 | 處理方式 |
|---------|---------|---------|
| prediction_result 缺少必要欄位 | ValueError | 拋出詳細錯誤訊息 |
| 輸出目錄不存在 | IOError | 嘗試建立目錄,失敗則拋出 |
| matplotlib 導入失敗 | ImportError | 拋出並建議安裝指令 |
| 中文字型缺失 | Warning | 記錄警告,降級使用預設字型 |

### 效能保證

- 執行時間: < 3 秒 (符合 spec SC-009)
- 記憶體使用: < 100 MB
- 檔案大小: < 500 KB (PNG, DPI=150)

### 檔案命名規則

若 `output_path` 為 None,自動生成檔名:
```
prediction_{input_date}_{prediction_date}.png
```
例如: `prediction_2024-02-15_2024-03-15.png`

### 測試案例

```python
# 測試案例 1: 基本繪圖
result = {
    "input_date": "2024-02-15",
    "prediction_date": "2024-03-15",
    "prob_5class": np.array([0.05, 0.15, 0.25, 0.35, 0.20]),
    "prob_3class": {"看跌": 0.20, "震盪": 0.25, "看漲": 0.55},
    "predicted_class_5": 3,
    "predicted_class_3": "看漲",
    "confidence": 0.35,
    "actual_data": None
}
path = plot_dual_view(result)
assert os.path.exists(path)
assert path.endswith(".png")

# 測試案例 2: 包含實際資料
result_with_actual = {**result, "actual_data": {...}}
path = plot_dual_view(result_with_actual, "outputs/test.png")
assert "正確✓" in get_image_text(path)  # OCR驗證
```

---

## 輔助函式: get_default_config

### 簽名

```python
def get_default_config() -> Dict[str, Any]:
    """
    取得預設視覺化配置

    Returns:
        Dict: VisualizationConfigDict (data-model.md定義)
    """
```

### 後置條件

- 回傳值包含 VisualizationConfigDict 所有欄位
- 所有值符合 data-model.md 定義的預設值

---

## 模組版本歷史

| 版本 | 日期 | 變更內容 |
|------|------|---------|
| 1.0.0 | 2025-11-16 | 初始版本,定義 plot_dual_view 與 get_default_config |
