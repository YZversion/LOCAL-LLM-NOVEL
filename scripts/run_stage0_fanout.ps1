# Stage 0 fan-out: 3 模型 × 4 anchor × 2 repeat = 24 格
# v2 × ch1_clean × s1101 已落盘（outputs/stage0/v2_ch1_clean_s1101.txt），跳过，实跑 23 格。
#
# 用法（在 repo 根目录，.venv-train 已激活）：
#   .venv-train\Scripts\Activate.ps1
#   .\scripts\run_stage0_fanout.ps1
#
# VRAM 安全门：每格每轮超 7.8 GB 自动中止该格，落盘已累积内容。
# 停止条件：累积新文本 >= 2300c（--target-chars 默认值）。
# reference 全程固定：data/raw/风丝引_原文.txt

$ErrorActionPreference = "Stop"
$anchors = @("ch1_clean", "ch58_bad_trigger", "mid_court_dialogue", "yehuan_controlled")
$seeds   = @(1101, 2202)

mkdir -Force outputs\stage0 | Out-Null

# ── 纯基座（不传 --adapter）──────────────────────────────────────────────────
# 需要 HF proxy 环境变量（离线解析 base model repo id）
$proxy = "$PSScriptRoot\..\outputs\hf_stage0_proxy"
$env:HF_HOME        = $proxy
$env:HF_HUB_CACHE   = "$proxy\hub"
$env:HF_XET_CACHE   = "$proxy\xet"

Write-Host "`n===== 纯基座 (8 格) =====" -ForegroundColor Cyan
foreach ($a in $anchors) {
    foreach ($s in $seeds) {
        $out = "outputs\stage0\base_${a}_s${s}.txt"
        if (Test-Path $out) {
            Write-Host "  [SKIP] 已存在: $out" -ForegroundColor DarkGray
            continue
        }
        Write-Host "  base x $a x s$s" -ForegroundColor Yellow
        python pipeline/adapter_cli.py `
            --raw-prompt-file "outputs\eval_anchors\$a.txt" `
            --max-seq-length 4096 --seed $s --batch `
            --out-file $out
        if ($LASTEXITCODE -ne 0) { Write-Warning "  [WARN] 退出码 $LASTEXITCODE，继续下格" }
    }
}

# 清空 HF proxy 变量，v2/v4 用本地路径不需要
Remove-Item Env:\HF_HOME        -ErrorAction SilentlyContinue
Remove-Item Env:\HF_HUB_CACHE   -ErrorAction SilentlyContinue
Remove-Item Env:\HF_XET_CACHE   -ErrorAction SilentlyContinue

# ── v2 adapter ───────────────────────────────────────────────────────────────
Write-Host "`n===== v2 adapter (8 格，跳 ch1_clean×s1101) =====" -ForegroundColor Cyan
foreach ($a in $anchors) {
    foreach ($s in $seeds) {
        if ($a -eq "ch1_clean" -and $s -eq 1101) {
            Write-Host "  [SKIP] v2 x ch1_clean x s1101 已落盘" -ForegroundColor DarkGray
            continue
        }
        $out = "outputs\stage0\v2_${a}_s${s}.txt"
        if (Test-Path $out) {
            Write-Host "  [SKIP] 已存在: $out" -ForegroundColor DarkGray
            continue
        }
        Write-Host "  v2 x $a x s$s" -ForegroundColor Yellow
        python pipeline/adapter_cli.py `
            --adapter outputs\qlora_run_v2\ `
            --raw-prompt-file "outputs\eval_anchors\$a.txt" `
            --max-seq-length 4096 --seed $s --batch `
            --out-file $out
        if ($LASTEXITCODE -ne 0) { Write-Warning "  [WARN] 退出码 $LASTEXITCODE，继续下格" }
    }
}

# ── v4 adapter ───────────────────────────────────────────────────────────────
Write-Host "`n===== v4 adapter (8 格) =====" -ForegroundColor Cyan
foreach ($a in $anchors) {
    foreach ($s in $seeds) {
        $out = "outputs\stage0\v4_${a}_s${s}.txt"
        if (Test-Path $out) {
            Write-Host "  [SKIP] 已存在: $out" -ForegroundColor DarkGray
            continue
        }
        Write-Host "  v4 x $a x s$s" -ForegroundColor Yellow
        python pipeline/adapter_cli.py `
            --adapter outputs\qlora_run_v4\ `
            --raw-prompt-file "outputs\eval_anchors\$a.txt" `
            --max-seq-length 4096 --seed $s --batch `
            --out-file $out
        if ($LASTEXITCODE -ne 0) { Write-Warning "  [WARN] 退出码 $LASTEXITCODE，继续下格" }
    }
}

Write-Host "`n===== 生成完成，开始评测 24 格 =====" -ForegroundColor Cyan

# ── 评测：扫所有 .txt，跳过已有 _eval.json 的格 ──────────────────────────────
Get-ChildItem outputs\stage0\*.txt | Where-Object { $_.Name -notlike "*_eval*" } |
ForEach-Object {
    $json = $_.FullName -replace '\.txt$', '_eval.json'
    if (Test-Path $json) {
        Write-Host "  [SKIP] eval 已存在: $($_.Name)" -ForegroundColor DarkGray
        return
    }
    Write-Host "  eval: $($_.Name)" -ForegroundColor Yellow
    python scripts/eval_draft.py `
        --candidate $_.FullName `
        --reference data/raw/风丝引_原文.txt `
        --config config.yaml `
        --out-json $json
    if ($LASTEXITCODE -ne 0) { Write-Warning "  [WARN] eval 失败: $($_.Name)" }
}

Write-Host "`n[DONE] Stage 0 fan-out + eval 全部完成。" -ForegroundColor Green
Write-Host "候选文件: outputs\stage0\*.txt"
Write-Host "评测结果: outputs\stage0\*_eval.json"
