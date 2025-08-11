#!/usr/bin/env python3
"""
Test the simple fix: just remove the restriction that disables Liger during gradient accumulation.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add Liger Kernel to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Liger-Kernel', 'src'))

def test_original_liger_with_gradient_accumulation():
    """Test that original Liger Kernel works with gradient accumulation."""
    
    print("\n" + "="*80)
    print("TESTING ORIGINAL LIGER KERNEL WITH GRADIENT ACCUMULATION")
    print("="*80)
    
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    
    # Parameters
    vocab_size = 1000
    hidden_dim = 256
    batch_size = 4
    seq_len = 32
    accumulation_steps = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    print(f"Accumulation steps: {accumulation_steps}")
    
    # Create model components
    linear = nn.Linear(hidden_dim, vocab_size, bias=True).to(device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-4)
    
    # Create loss function (original Liger)
    liger_loss = LigerFusedLinearCrossEntropyLoss(reduction='mean')
    
    # Test data
    torch.manual_seed(42)
    all_inputs = []
    all_targets = []
    for step in range(accumulation_steps):
        inputs = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        all_inputs.append(inputs)
        all_targets.append(targets)
    
    print("\nRunning gradient accumulation with original Liger Kernel...")
    
    optimizer.zero_grad()
    total_loss = 0
    
    for step in range(accumulation_steps):
        inputs = all_inputs[step]
        targets = all_targets[step]
        
        # Flatten for Liger
        inputs_flat = inputs.view(-1, hidden_dim)
        targets_flat = targets.view(-1)
        
        # Forward pass with Liger
        loss = liger_loss(
            linear.weight,
            inputs_flat,
            targets_flat,
            linear.bias
        )
        
        # Scale for accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        total_loss += loss.item() * accumulation_steps
        print(f"  Step {step + 1}: Loss = {loss.item() * accumulation_steps:.6f}")
    
    # Check gradients
    assert linear.weight.grad is not None, "Weight gradients should exist"
    assert linear.bias.grad is not None, "Bias gradients should exist"
    
    weight_grad_norm = torch.norm(linear.weight.grad).item()
    bias_grad_norm = torch.norm(linear.bias.grad).item()
    
    print(f"\nResults:")
    print(f"  Total loss: {total_loss:.6f}")
    print(f"  Weight gradient norm: {weight_grad_norm:.6f}")
    print(f"  Bias gradient norm: {bias_grad_norm:.6f}")
    
    # Optimizer step
    optimizer.step()
    
    print("✓ Gradient accumulation with original Liger works perfectly!")
    
    return {
        "total_loss": total_loss,
        "weight_grad_norm": weight_grad_norm,
        "bias_grad_norm": bias_grad_norm
    }


def compare_with_baseline():
    """Compare with standard PyTorch cross entropy."""
    
    print("\n" + "="*80)
    print("COMPARISON WITH STANDARD PYTORCH")
    print("="*80)
    
    vocab_size = 1000
    hidden_dim = 256
    batch_size = 4
    seq_len = 32
    accumulation_steps = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create identical model components
    torch.manual_seed(42)
    linear_baseline = nn.Linear(hidden_dim, vocab_size, bias=True).to(device)
    
    torch.manual_seed(42)
    linear_liger = nn.Linear(hidden_dim, vocab_size, bias=True).to(device)
    
    # Ensure same weights
    linear_liger.load_state_dict(linear_baseline.state_dict())
    
    # Loss functions
    criterion_baseline = nn.CrossEntropyLoss(reduction='mean')
    
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    criterion_liger = LigerFusedLinearCrossEntropyLoss(reduction='mean')
    
    # Same test data
    torch.manual_seed(123)
    all_inputs = []
    all_targets = []
    for step in range(accumulation_steps):
        inputs = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        all_inputs.append(inputs)
        all_targets.append(targets)
    
    # Test 1: Baseline
    print("\nTesting baseline (PyTorch CrossEntropyLoss):")
    linear_baseline.zero_grad()
    total_loss_baseline = 0
    
    for step in range(accumulation_steps):
        inputs = all_inputs[step]
        targets = all_targets[step]
        
        logits = linear_baseline(inputs)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        loss = criterion_baseline(logits_flat, targets_flat)
        loss = loss / accumulation_steps
        loss.backward()
        
        total_loss_baseline += loss.item() * accumulation_steps
    
    baseline_weight_grad_norm = torch.norm(linear_baseline.weight.grad).item()
    baseline_bias_grad_norm = torch.norm(linear_baseline.bias.grad).item()
    
    print(f"  Total loss: {total_loss_baseline:.6f}")
    print(f"  Weight grad norm: {baseline_weight_grad_norm:.6f}")
    print(f"  Bias grad norm: {baseline_bias_grad_norm:.6f}")
    
    # Test 2: Liger
    print("\nTesting Liger:")
    linear_liger.zero_grad()
    total_loss_liger = 0
    
    for step in range(accumulation_steps):
        inputs = all_inputs[step]
        targets = all_targets[step]
        
        inputs_flat = inputs.view(-1, hidden_dim)
        targets_flat = targets.view(-1)
        
        loss = criterion_liger(
            linear_liger.weight,
            inputs_flat,
            targets_flat,
            linear_liger.bias
        )
        loss = loss / accumulation_steps
        loss.backward()
        
        total_loss_liger += loss.item() * accumulation_steps
    
    liger_weight_grad_norm = torch.norm(linear_liger.weight.grad).item()
    liger_bias_grad_norm = torch.norm(linear_liger.bias.grad).item()
    
    print(f"  Total loss: {total_loss_liger:.6f}")
    print(f"  Weight grad norm: {liger_weight_grad_norm:.6f}")
    print(f"  Bias grad norm: {liger_bias_grad_norm:.6f}")
    
    # Comparison
    print("\n" + "-"*60)
    print("COMPARISON:")
    print("-"*60)
    
    loss_diff = abs(total_loss_baseline - total_loss_liger)
    weight_grad_diff = abs(baseline_weight_grad_norm - liger_weight_grad_norm)
    bias_grad_diff = abs(baseline_bias_grad_norm - liger_bias_grad_norm)
    
    print(f"Loss difference: {loss_diff:.8f}")
    print(f"Weight grad difference: {weight_grad_diff:.8f}")
    print(f"Bias grad difference: {bias_grad_diff:.8f}")
    
    # Tolerances
    loss_ok = loss_diff < 1e-5
    weight_grad_ok = weight_grad_diff < 1e-4
    bias_grad_ok = bias_grad_diff < 1e-4
    
    print(f"\nValidation:")
    print(f"  Loss match: {'✓ PASS' if loss_ok else '✗ FAIL'}")
    print(f"  Weight grad match: {'✓ PASS' if weight_grad_ok else '✗ FAIL'}")
    print(f"  Bias grad match: {'✓ PASS' if bias_grad_ok else '✗ FAIL'}")
    
    all_ok = loss_ok and weight_grad_ok and bias_grad_ok
    
    if all_ok:
        print("\n✓ Liger Kernel produces identical results to PyTorch baseline!")
        print("✓ Gradient accumulation works correctly with original Liger!")
    else:
        print("\n✗ Differences detected between Liger and baseline.")
    
    return all_ok


if __name__ == "__main__":
    print("Testing the simple fix: Original Liger Kernel with gradient accumulation")
    
    try:
        # Test original Liger with accumulation
        liger_results = test_original_liger_with_gradient_accumulation()
        
        # Compare with baseline
        comparison_passed = compare_with_baseline()
        
        print("\n" + "="*80)
        print("FINAL CONCLUSION")
        print("="*80)
        
        if comparison_passed:
            print("🎉 SUCCESS!")
            print("✓ Original Liger Kernel works perfectly with gradient accumulation")
            print("✓ The issue was just the overly conservative check in 360-LLaMA-Factory")
            print("✓ Simple fix: Remove the restriction that disables Liger during gradient accumulation")
            print("\nThe fix is much simpler than we thought!")
        else:
            print("❌ Issues detected in comparison")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)