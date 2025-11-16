#!/bin/bash
# 完整的訓練與超參數調整工作流程範例
# 這個腳本展示如何從基準模型訓練到超參數調整的完整流程

echo "================================================================================"
echo "LSTM 台股預測系統 - 完整工作流程範例"
echo "================================================================================"
echo ""

# 設定變數
DATA_FILE="19980601-20251111-converted.csv"
FEATURE_SET="Set A"
TIME_STEPS=60
EPOCHS=100
BATCH_SIZE=32
MAX_TRIALS=10  # 為了演示使用較少試驗次數，實際應用建議 50-100
EPOCHS_PER_TRIAL=30

echo "📋 配置參數:"
echo "  資料檔案: $DATA_FILE"
echo "  特徵集: $FEATURE_SET"
echo "  時間窗口: $TIME_STEPS"
echo "  訓練週期: $EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  超參數試驗次數: $MAX_TRIALS"
echo ""

# ============================================================================
# 步驟 1: 訓練基準模型
# ============================================================================
echo "================================================================================"
echo "步驟 1/3: 訓練基準模型"
echo "================================================================================"
echo ""

python src/cli/train.py \
    --data-file "$DATA_FILE" \
    --feature-set "$FEATURE_SET" \
    --time-steps $TIME_STEPS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --model-name baseline_model

if [ $? -ne 0 ]; then
    echo "❌ 基準模型訓練失敗"
    exit 1
fi

echo ""
echo "✅ 基準模型訓練完成"
echo ""

# 找到最新的基準模型
BASELINE_MODEL=$(ls -t models/baseline_model_*.h5 2>/dev/null | head -1)

if [ -z "$BASELINE_MODEL" ]; then
    echo "❌ 找不到基準模型檔案"
    exit 1
fi

echo "📊 基準模型: $BASELINE_MODEL"
echo ""

# ============================================================================
# 步驟 2: 執行超參數調整（與基準模型比較）
# ============================================================================
echo "================================================================================"
echo "步驟 2/3: 執行超參數調整（與基準模型比較）"
echo "================================================================================"
echo ""

python src/cli/tune.py \
    --data-file "$DATA_FILE" \
    --max-trials $MAX_TRIALS \
    --epochs-per-trial $EPOCHS_PER_TRIAL \
    --baseline-model "$BASELINE_MODEL" \
    --tuned-model-name best_tuned_model \
    --tuner-type random \
    --output-dir models/

if [ $? -ne 0 ]; then
    echo "❌ 超參數調整失敗"
    exit 1
fi

echo ""
echo "✅ 超參數調整完成"
echo ""

# ============================================================================
# 步驟 3: 查看結果
# ============================================================================
echo "================================================================================"
echo "步驟 3/3: 查看結果"
echo "================================================================================"
echo ""

# 找到最新的調整模型和報告
TUNED_MODEL=$(ls -t models/best_tuned_model_*.h5 2>/dev/null | head -1)
COMPARISON_REPORT=$(ls -t logs/tuning_logs/comparison_report_*.txt 2>/dev/null | head -1)
TUNING_RESULTS=$(ls -t logs/tuning_logs/tuning_results_*.txt 2>/dev/null | head -1)

if [ -z "$TUNED_MODEL" ]; then
    echo "❌ 找不到調整模型檔案"
    exit 1
fi

echo "📊 生成的檔案:"
echo "  基準模型: $BASELINE_MODEL"
echo "  調整模型: $TUNED_MODEL"
echo ""

if [ ! -z "$COMPARISON_REPORT" ]; then
    echo "  比較報告: $COMPARISON_REPORT"
    echo ""
    echo "-------------------------------------------------------------------------------"
    echo "比較報告內容:"
    echo "-------------------------------------------------------------------------------"
    cat "$COMPARISON_REPORT"
    echo ""
fi

if [ ! -z "$TUNING_RESULTS" ]; then
    echo "  調整結果: $TUNING_RESULTS"
    echo ""
    echo "-------------------------------------------------------------------------------"
    echo "調整結果摘要:"
    echo "-------------------------------------------------------------------------------"
    head -50 "$TUNING_RESULTS"
    echo ""
    echo "(完整結果請查看: $TUNING_RESULTS)"
    echo ""
fi

# ============================================================================
# 完成
# ============================================================================
echo "================================================================================"
echo "✅ 工作流程完成！"
echo "================================================================================"
echo ""
echo "下一步建議:"
echo "  1. 查看完整的比較報告: cat $COMPARISON_REPORT"
echo "  2. 查看完整的調整結果: cat $TUNING_RESULTS"
echo "  3. 查看模型配置: cat ${TUNED_MODEL%.h5}_config.json"
echo "  4. 如果調整模型優於基準模型，可以使用它進行預測"
echo "  5. 如果需要進一步改善，可以增加試驗次數重新調整"
echo ""
echo "================================================================================"
