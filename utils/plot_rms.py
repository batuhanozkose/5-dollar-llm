import matplotlib.pyplot as plt
import json
import argparse
import os
import math

def plot_weight_rms(metrics_file, output_file, eta, lam):
    if not os.path.exists(metrics_file):
        print(f"Error: File not found {metrics_file}")
        return

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    history = data.get('history', {})
    steps = history.get('steps', [])
    rms = history.get('weight_rms', [])
    
    if not steps or not rms:
        print("Error: Weight RMS data not found in history")
        return

    theoretical = math.sqrt(eta / (2 * lam))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, rms, marker='o', linestyle='-', color='purple', label='Actual Weight RMS')
    plt.axhline(y=theoretical, color='r', linestyle='--', label=f'Theoretical Limit ({theoretical:.4f})')
    
    plt.title(f"Weight RMS Equilibrium Verification (Eq 238)\n$\eta$={eta}, $\lambda$={lam}")
    plt.xlabel("Steps")
    plt.ylabel("Weight RMS")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    for i, (x, y) in enumerate(zip(steps, rms)):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot weight RMS vs theoretical limit")
    parser.add_argument("metrics_file", help="Path to metrics.json")
    parser.add_argument("output_file", help="Path to save the plot image")
    parser.add_argument("--eta", type=float, default=0.02, help="Learning rate used")
    parser.add_argument("--lam", type=float, default=0.2, help="Weight decay used")
    
    args = parser.parse_args()
    plot_weight_rms(args.metrics_file, args.output_file, args.eta, args.lam)
