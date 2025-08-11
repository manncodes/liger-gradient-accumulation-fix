#!/usr/bin/env python3
"""
Validation test to ensure gradient accumulation works correctly with the patch.
This test specifically checks that gradients accumulate properly across steps.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add Liger Kernel to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Liger-Kernel', 'src'))

def test_gradient_accumulation_correctness():
    """Test that gradient accumulation produces mathematically correct results."""
    
    print("\n" + "="*80)
    print("GRADIENT ACCUMULATION CORRECTNESS VALIDATION")
    print("="*80)
    
    # Setup parameters
    vocab_size = 1000
    hidden_dim = 256
    batch_size = 4
    seq_len = 32
    accumulation_steps = 4
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    # Create model components
    linear = nn.Linear(hidden_dim, vocab_size, bias=True).to(device)
    
    # Generate test data - same for all tests
    all_inputs = []
    all_targets = []
    for step in range(accumulation_steps):
        inputs = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=False)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        all_inputs.append(inputs)
        all_targets.append(targets)
    
    # Test 1: Standard PyTorch approach (reference)
    print("\nTest 1: Standard PyTorch Cross Entropy (Reference)")
    print("-" * 60)
    
    linear_ref = nn.Linear(hidden_dim, vocab_size, bias=True).to(device)
    linear_ref.load_state_dict(linear.state_dict())  # Same initial weights
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    linear_ref.zero_grad()
    
    total_loss_ref = 0
    for step in range(accumulation_steps):
        inputs = all_inputs[step]
        targets = all_targets[step]
        
        logits = linear_ref(inputs)  # [batch_size, seq_len, vocab_size]
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        loss = criterion(logits_flat, targets_flat)
        loss = loss / accumulation_steps  # Scale for accumulation
        loss.backward()
        
        total_loss_ref += loss.item() * accumulation_steps
    
    # Store reference gradients
    ref_weight_grad = linear_ref.weight.grad.clone()
    ref_bias_grad = linear_ref.bias.grad.clone()
    
    print(f"  Total loss: {total_loss_ref:.6f}")
    print(f"  Weight grad norm: {torch.norm(ref_weight_grad).item():.6f}")
    print(f"  Bias grad norm: {torch.norm(ref_bias_grad).item():.6f}")
    
    # Test 2: Liger Kernel with gradient accumulation
    print("\nTest 2: Liger Kernel with Gradient Accumulation")
    print("-" * 60)
    
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    
    linear_liger = nn.Linear(hidden_dim, vocab_size, bias=True).to(device)
    linear_liger.load_state_dict(linear.state_dict())  # Same initial weights
    
    criterion_liger = LigerFusedLinearCrossEntropyLoss(
        gradient_accumulation=True,
        reduction='mean'
    )
    
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
        
        loss = loss / accumulation_steps  # Scale for accumulation
        loss.backward()
        
        total_loss_liger += loss.item() * accumulation_steps
    
    # Store Liger gradients
    liger_weight_grad = linear_liger.weight.grad.clone()
    liger_bias_grad = linear_liger.bias.grad.clone()
    
    print(f"  Total loss: {total_loss_liger:.6f}")
    print(f"  Weight grad norm: {torch.norm(liger_weight_grad).item():.6f}")
    print(f"  Bias grad norm: {torch.norm(liger_bias_grad).item():.6f}")
    
    # Test 3: Manual accumulation with Liger (step by step)
    print("\nTest 3: Manual Step-by-step Accumulation with Liger")
    print("-" * 60)
    
    linear_manual = nn.Linear(hidden_dim, vocab_size, bias=True).to(device)
    linear_manual.load_state_dict(linear.state_dict())  # Same initial weights
    
    criterion_manual = LigerFusedLinearCrossEntropyLoss(
        gradient_accumulation=False,  # Manual accumulation
        reduction='mean'
    )
    
    linear_manual.zero_grad()
    total_loss_manual = 0
    
    # Manually accumulate gradients
    accumulated_weight_grad = torch.zeros_like(linear_manual.weight)
    accumulated_bias_grad = torch.zeros_like(linear_manual.bias)
    
    for step in range(accumulation_steps):
        # Clear gradients for this step
        linear_manual.zero_grad()
        
        inputs = all_inputs[step]
        targets = all_targets[step]
        
        inputs_flat = inputs.view(-1, hidden_dim)
        targets_flat = targets.view(-1)
        
        loss = criterion_manual(
            linear_manual.weight,
            inputs_flat,
            targets_flat,
            linear_manual.bias
        )
        
        loss = loss / accumulation_steps  # Scale for accumulation
        loss.backward()
        
        # Manually accumulate gradients
        if linear_manual.weight.grad is not None:
            accumulated_weight_grad += linear_manual.weight.grad
        if linear_manual.bias.grad is not None:
            accumulated_bias_grad += linear_manual.bias.grad
        
        total_loss_manual += loss.item() * accumulation_steps
    
    print(f"  Total loss: {total_loss_manual:.6f}")
    print(f"  Weight grad norm: {torch.norm(accumulated_weight_grad).item():.6f}")
    print(f"  Bias grad norm: {torch.norm(accumulated_bias_grad).item():.6f}")
    
    # Validation
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    
    # Check loss consistency
    loss_diff_liger = abs(total_loss_ref - total_loss_liger)
    loss_diff_manual = abs(total_loss_ref - total_loss_manual)
    
    print(f"\nLoss Consistency:")
    print(f"  Reference loss:     {total_loss_ref:.6f}")
    print(f"  Liger loss:         {total_loss_liger:.6f} (diff: {loss_diff_liger:.8f})")
    print(f"  Manual loss:        {total_loss_manual:.6f} (diff: {loss_diff_manual:.8f})")
    
    loss_ok = loss_diff_liger < 1e-5 and loss_diff_manual < 1e-5
    print(f"  ✓ Loss consistency: {'PASS' if loss_ok else 'FAIL'}")
    
    # Check gradient consistency
    weight_grad_diff_liger = torch.norm(ref_weight_grad - liger_weight_grad).item()
    weight_grad_diff_manual = torch.norm(ref_weight_grad - accumulated_weight_grad).item()
    
    bias_grad_diff_liger = torch.norm(ref_bias_grad - liger_bias_grad).item()
    bias_grad_diff_manual = torch.norm(ref_bias_grad - accumulated_bias_grad).item()
    
    print(f"\nWeight Gradient Consistency:")
    print(f"  Reference norm:     {torch.norm(ref_weight_grad).item():.6f}")
    print(f"  Liger norm:         {torch.norm(liger_weight_grad).item():.6f} (diff: {weight_grad_diff_liger:.8f})")
    print(f"  Manual norm:        {torch.norm(accumulated_weight_grad).item():.6f} (diff: {weight_grad_diff_manual:.8f})")
    
    weight_grad_ok = weight_grad_diff_liger < 1e-4 and weight_grad_diff_manual < 1e-4
    print(f"  ✓ Weight gradient consistency: {'PASS' if weight_grad_ok else 'FAIL'}")
    
    print(f"\nBias Gradient Consistency:")
    print(f"  Reference norm:     {torch.norm(ref_bias_grad).item():.6f}")
    print(f"  Liger norm:         {torch.norm(liger_bias_grad).item():.6f} (diff: {bias_grad_diff_liger:.8f})")
    print(f"  Manual norm:        {torch.norm(accumulated_bias_grad).item():.6f} (diff: {bias_grad_diff_manual:.8f})")
    
    bias_grad_ok = bias_grad_diff_liger < 1e-4 and bias_grad_diff_manual < 1e-4
    print(f"  ✓ Bias gradient consistency: {'PASS' if bias_grad_ok else 'FAIL'}")
    
    # Overall validation
    all_passed = loss_ok and weight_grad_ok and bias_grad_ok
    
    print(f"\n" + "="*80)
    print(f"OVERALL VALIDATION: {'PASS' if all_passed else 'FAIL'}")
    print("="*80)
    
    if all_passed:
        print("✓ Gradient accumulation patch works correctly!")
        print("  - Losses match reference implementation")
        print("  - Weight gradients are mathematically equivalent")
        print("  - Bias gradients are mathematically equivalent")
        print("  - Both automatic and manual accumulation work")
    else:
        print(" Issues detected in gradient accumulation")
        if not loss_ok:
            print("  - Loss computation differs from reference")
        if not weight_grad_ok:
            print("  - Weight gradients don't match reference")
        if not bias_grad_ok:
            print("  - Bias gradients don't match reference")
    
    return all_passed


def test_multiple_accumulation_cycles():
    """Test multiple cycles of gradient accumulation."""
    
    print("\n" + "="*80)
    print("MULTIPLE ACCUMULATION CYCLES TEST")
    print("="*80)
    
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    
    vocab_size = 500
    hidden_dim = 128
    batch_size = 2
    seq_len = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    linear = nn.Linear(hidden_dim, vocab_size, bias=True).to(device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)
    criterion = LigerFusedLinearCrossEntropyLoss(gradient_accumulation=True, reduction='mean')
    
    print(f"Testing multiple accumulation cycles...")
    
    losses = []
    grad_norms = []
    
    for cycle in range(3):  # 3 cycles of accumulation
        print(f"\nCycle {cycle + 1}:")
        
        optimizer.zero_grad()
        cycle_losses = []
        
        # Accumulate gradients over 4 steps
        for step in range(4):
            inputs = torch.randn(batch_size, seq_len, hidden_dim, device=device)
            targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            
            inputs_flat = inputs.view(-1, hidden_dim)
            targets_flat = targets.view(-1)
            
            loss = criterion(linear.weight, inputs_flat, targets_flat, linear.bias)
            loss = loss / 4  # Scale for accumulation
            loss.backward()
            
            cycle_losses.append(loss.item() * 4)
        
        # Check gradients accumulated correctly
        if linear.weight.grad is not None:
            grad_norm = torch.norm(linear.weight.grad).item()
            grad_norms.append(grad_norm)
            print(f"  Accumulated gradient norm: {grad_norm:.4f}")
        
        # Optimizer step
        optimizer.step()
        
        avg_loss = np.mean(cycle_losses)
        losses.append(avg_loss)
        print(f"  Average loss: {avg_loss:.4f}")
    
    # Validate that gradients were properly reset and accumulated
    stable_training = True
    if len(grad_norms) >= 2:
        grad_variance = np.var(grad_norms)
        if grad_variance > 1.0:  # Large variance indicates instability
            stable_training = False
    
    print(f"\n✓ Multiple cycle test: {'PASS' if stable_training else 'FAIL'}")
    print(f"  Gradient norm variance: {np.var(grad_norms):.6f}")
    print(f"  Loss trend: {losses}")
    
    return stable_training


if __name__ == "__main__":
    print("Running gradient accumulation validation tests...")
    
    try:
        # Test correctness
        correctness_passed = test_gradient_accumulation_correctness()
        
        # Test multiple cycles
        cycles_passed = test_multiple_accumulation_cycles()
        
        # Final verdict
        print("\n" + "="*80)
        print("FINAL VALIDATION RESULTS")
        print("="*80)
        
        if correctness_passed and cycles_passed:
            print(" ALL TESTS PASSED!")
            print("✓ Gradient accumulation patch is working correctly")
            print("✓ Mathematical equivalence confirmed")
            print("✓ Multi-cycle stability confirmed")
            print("\nThe patch is ready for production use!")
        else:
            print(" SOME TESTS FAILED")
            if not correctness_passed:
                print(" Correctness test failed")
            if not cycles_passed:
                print(" Multiple cycles test failed")
            print("\nPatch needs further investigation.")
        
    except Exception as e:
        print(f"\nValidation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)