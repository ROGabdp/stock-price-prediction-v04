@echo off
REM 完整的訓練與超參數調整工作流程範例（Windows 版本）
REM 這個腳本展示如何從基準模型訓練到超參數調整的完整流程

chcp 65001 > nul
setlocal enabledelayedexpansion

echo ================================================================================
echo LSTM 台股預測系統 - 完整工作流程範例
echo ================================================================================
echo.

REM 設定變數
set DATA_FILE=19980601-20251111-converted.csv
set FEATURE_SET=Set A
set TIME_STEPS=60
set EPOCHS=100
set BATCH_SIZE=32
set MAX_TRIALS=10
set EPOCHS_PER_TRIAL=30

echo 配置參數:
echo   資料檔案: %DATA_FILE%
echo   特徵集: %FEATURE_SET%
echo   時間窗口: %TIME_STEPS%
echo   訓練週期: %EPOCHS%
echo   批次大小: %BATCH_SIZE%
echo   超參數試驗次數: %MAX_TRIALS%
echo.

REM ============================================================================
REM 步驟 1: 訓練基準模型
REM ============================================================================
echo ================================================================================
echo 步驟 1/3: 訓練基準模型
echo ================================================================================
echo.

python src\cli\train.py ^
    --data-file "%DATA_FILE%" ^
    --feature-set "%FEATURE_SET%" ^
    --time-steps %TIME_STEPS% ^
    --epochs %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --model-name baseline_model

if %ERRORLEVEL% neq 0 (
    echo 基準模型訓練失敗
    exit /b 1
)

echo.
echo 基準模型訓練完成
echo.

REM 找到最新的基準模型
for /f "delims=" %%i in ('dir /b /o-d models\baseline_model_*.h5 2^>nul') do (
    set BASELINE_MODEL=models\%%i
    goto :found_baseline
)

:found_baseline
if "%BASELINE_MODEL%"=="" (
    echo 找不到基準模型檔案
    exit /b 1
)

echo 基準模型: %BASELINE_MODEL%
echo.

REM ============================================================================
REM 步驟 2: 執行超參數調整（與基準模型比較）
REM ============================================================================
echo ================================================================================
echo 步驟 2/3: 執行超參數調整（與基準模型比較）
echo ================================================================================
echo.

python src\cli\tune.py ^
    --data-file "%DATA_FILE%" ^
    --max-trials %MAX_TRIALS% ^
    --epochs-per-trial %EPOCHS_PER_TRIAL% ^
    --baseline-model "%BASELINE_MODEL%" ^
    --tuned-model-name best_tuned_model ^
    --tuner-type random ^
    --output-dir models/

if %ERRORLEVEL% neq 0 (
    echo 超參數調整失敗
    exit /b 1
)

echo.
echo 超參數調整完成
echo.

REM ============================================================================
REM 步驟 3: 查看結果
REM ============================================================================
echo ================================================================================
echo 步驟 3/3: 查看結果
echo ================================================================================
echo.

REM 找到最新的調整模型和報告
for /f "delims=" %%i in ('dir /b /o-d models\best_tuned_model_*.h5 2^>nul') do (
    set TUNED_MODEL=models\%%i
    goto :found_tuned
)

:found_tuned
for /f "delims=" %%i in ('dir /b /o-d logs\tuning_logs\comparison_report_*.txt 2^>nul') do (
    set COMPARISON_REPORT=logs\tuning_logs\%%i
    goto :found_comparison
)

:found_comparison
for /f "delims=" %%i in ('dir /b /o-d logs\tuning_logs\tuning_results_*.txt 2^>nul') do (
    set TUNING_RESULTS=logs\tuning_logs\%%i
    goto :found_results
)

:found_results
if "%TUNED_MODEL%"=="" (
    echo 找不到調整模型檔案
    exit /b 1
)

echo 生成的檔案:
echo   基準模型: %BASELINE_MODEL%
echo   調整模型: %TUNED_MODEL%
echo.

if not "%COMPARISON_REPORT%"=="" (
    echo   比較報告: %COMPARISON_REPORT%
    echo.
    echo -------------------------------------------------------------------------------
    echo 比較報告內容:
    echo -------------------------------------------------------------------------------
    type "%COMPARISON_REPORT%"
    echo.
)

if not "%TUNING_RESULTS%"=="" (
    echo   調整結果: %TUNING_RESULTS%
    echo.
    echo -------------------------------------------------------------------------------
    echo 調整結果摘要:
    echo -------------------------------------------------------------------------------
    powershell -Command "Get-Content '%TUNING_RESULTS%' | Select-Object -First 50"
    echo.
    echo (完整結果請查看: %TUNING_RESULTS%)
    echo.
)

REM ============================================================================
REM 完成
REM ============================================================================
echo ================================================================================
echo 工作流程完成！
echo ================================================================================
echo.
echo 下一步建議:
echo   1. 查看完整的比較報告: type "%COMPARISON_REPORT%"
echo   2. 查看完整的調整結果: type "%TUNING_RESULTS%"
echo   3. 如果調整模型優於基準模型，可以使用它進行預測
echo   4. 如果需要進一步改善，可以增加試驗次數重新調整
echo.
echo ================================================================================

endlocal
