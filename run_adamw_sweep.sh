#!/bin/bash

# run_adamw_sweep.sh
# Tests Optimizer Universality by running the same schedules with pure AdamW

TOKENS=8000000
BASE_DIR="checkpoints/adamw_sweep"

echo "🚀 Starting AdamW Universality Sweep..."
echo "This will test if the Inverse Sqrt schedule is optimal for AdamW alone."

# Experiment 1: AdamW Baseline (Constant)
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Experiment 1: AdamW Baseline (Constant LR/WD)"
echo "═══════════════════════════════════════════════════════════"
python train_llm.py \
    --schedule_type constant \
    --train_tokens $TOKENS \
    --weight_decay 0.2 \
    --use_adamw_only true \
    --output_dir ${BASE_DIR}/exp1_adamw_baseline

# Experiment 2: AdamW + Inverse Sqrt (Coupled, WD=0.2)
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Experiment 2: AdamW + Inverse Sqrt (Coupled, WD=0.2)"
echo "═══════════════════════════════════════════════════════════"
python train_llm.py \
    --schedule_type inverse_sqrt \
    --train_tokens $TOKENS \
    --weight_decay 0.2 \
    --coupled_wd true \
    --use_adamw_only true \
    --output_dir ${BASE_DIR}/exp2_adamw_sqrt_coupled

# Experiment 3: AdamW + Inverse Sqrt (Coupled, WD=0.4)
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Experiment 3: AdamW + Inverse Sqrt (Coupled, WD=0.4)"
echo "═══════════════════════════════════════════════════════════"
python train_llm.py \
    --schedule_type inverse_sqrt \
    --train_tokens $TOKENS \
    --weight_decay 0.4 \
    --coupled_wd true \
    --use_adamw_only true \
    --output_dir ${BASE_DIR}/exp3_adamw_sqrt_wd0.4

echo ""
echo "✅ AdamW Universality Sweep Complete!"
