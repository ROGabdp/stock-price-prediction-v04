# API Contracts: Enhanced Stock Price Prediction Visualization

**Feature**: 002-enhanced-prediction-viz
**Date**: 2025-11-16
**Purpose**: 定義模組之間的函式介面契約,確保整合一致性

## 概述

本功能為 CLI 工具,不涉及 REST API 或 GraphQL。此處定義的契約為 Python 模組之間的函式介面規範。

## 契約文件清單

1. [aggregator_contract.md](aggregator_contract.md) - 3分類聚合模組契約
2. [validator_contract.md](validator_contract.md) - 歷史驗證模組契約
3. [plotter_contract.md](plotter_contract.md) - 視覺化繪圖模組契約
4. [cli_contract.md](cli_contract.md) - CLI 入口點契約

## 模組依賴關係

```
src/cli/predict_enhanced.py (CLI入口)
    │
    ├──► src/prediction/predictor.py (現有,提供5分類機率)
    │
    ├──► src/visualization/aggregator.py (新增,3分類聚合)
    │
    ├──► src/visualization/validator.py (新增,歷史驗證)
    │
    └──► src/visualization/plotter.py (新增,視覺化繪圖)
```

## 介面穩定性承諾

- **Stable (穩定)**: 承諾不在無預警情況下變更簽名
- **Experimental (實驗性)**: 可能在未來版本變更
- **Deprecated (棄用)**: 將在未來版本移除

所有契約目前均為 **Stable** 狀態。

## 版本管理

契約文件使用語義化版本 (Semantic Versioning):
- **MAJOR**: 破壞性變更 (移除參數、變更回傳型別)
- **MINOR**: 向後相容的新增 (新增選用參數、新增函式)
- **PATCH**: 向後相容的修正 (文件更新、範例補充)

當前版本: **1.0.0**
