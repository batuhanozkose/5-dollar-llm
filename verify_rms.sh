#!/bin/bash

# verify_rms.sh
# Verifies Eq 238: Weight RMS ≈ sqrt(eta / 2lambda)

TOKENS=8000000
ETA=0.02
LAMBDA=0.2
THEORETICAL=$(echo "scale=4; sqrt($ETA / (2 * $LAMBDA))" | bc -l)

echo "🚀 Starting Weight RMS Verification Experiment..."
echo "📊 Theoretical Equilibrium RMS: $THEORETICAL"

python train_llm.py \
    --schedule_type constant \
    --train_tokens $TOKENS \
    --weight_decay $LAMBDA \
    --output_dir checkpoints/rms_verification

echo "✅ Experiment Complete."
