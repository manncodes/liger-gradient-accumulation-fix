#!/usr/bin/env python3
"""
Direct stability and performance comparison test for Liger Kernel with gradient accumulation.
Tests training stability without requiring full 360-LLaMA-Factory setup.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import json
from typing import Dict, List, Tuple
import sys
import os

# Add Liger Kernel to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Liger-Kernel', 'src'))


class DummyTextDataset(Dataset):
    """Simple dataset for testing."""
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random input and target sequences
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = torch.randint(0, self.vocab_size, (self.seq_len,))
        return input_ids, labels


class SimpleTransformerModel(nn.Module):
    """Simple transformer model for testing."""
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        return self.output(x)


def train_with_liger(
    model: nn.Module,
    dataloader: DataLoader,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 4,
    device: str = "cuda"
) -> Dict:
    """Train model using Liger Kernel's fused cross entropy."""
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Use Liger's fused cross entropy with gradient accumulation
    criterion = LigerFusedLinearCrossEntropyLoss(
        gradient_accumulation=True,
        reduction="mean"
    )
    
    losses = []
    grad_norms = []
    step_times = []
    memory_usage = []
    
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            step_start = time.time()
            
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward pass through embedding and transformer
            hidden_states = model.embedding(input_ids)
            hidden_states = model.transformer(hidden_states)
            
            # Flatten for cross entropy
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_states_flat = hidden_states.view(-1, hidden_dim)
            labels_flat = labels.view(-1)
            
            # Use Liger's fused linear cross entropy
            loss = criterion(
                model.output.weight,
                hidden_states_flat,
                labels_flat,
                model.output.bias
            )
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Track metrics
            epoch_losses.append(loss.item() * gradient_accumulation_steps)
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Calculate gradient norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                grad_norms.append(total_norm)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Track memory
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.memory_allocated() / 1024**3)
            
            step_times.append(time.time() - step_start)
            
            if batch_idx >= 20:  # Limit steps for testing
                break
        
        losses.extend(epoch_losses)
    
    total_time = time.time() - start_time
    
    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "step_times": step_times,
        "memory_usage": memory_usage,
        "total_time": total_time,
        "avg_loss": np.mean(losses),
        "loss_std": np.std(losses),
        "avg_grad_norm": np.mean(grad_norms),
        "grad_norm_std": np.std(grad_norms),
        "peak_memory_gb": max(memory_usage) if memory_usage else 0,
    }


def train_baseline(
    model: nn.Module,
    dataloader: DataLoader,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 4,
    device: str = "cuda"
) -> Dict:
    """Train model using standard PyTorch cross entropy."""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    grad_norms = []
    step_times = []
    memory_usage = []
    
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            step_start = time.time()
            
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Flatten for cross entropy
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            
            # Calculate loss
            loss = criterion(logits_flat, labels_flat)
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Track metrics
            epoch_losses.append(loss.item() * gradient_accumulation_steps)
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Calculate gradient norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                grad_norms.append(total_norm)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Track memory
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.memory_allocated() / 1024**3)
            
            step_times.append(time.time() - step_start)
            
            if batch_idx >= 20:  # Limit steps for testing
                break
        
        losses.extend(epoch_losses)
    
    total_time = time.time() - start_time
    
    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "step_times": step_times,
        "memory_usage": memory_usage,
        "total_time": total_time,
        "avg_loss": np.mean(losses),
        "loss_std": np.std(losses),
        "avg_grad_norm": np.mean(grad_norms),
        "grad_norm_std": np.std(grad_norms),
        "peak_memory_gb": max(memory_usage) if memory_usage else 0,
    }


def run_stability_comparison():
    """Run comprehensive stability comparison."""
    
    print("\n" + "="*80)
    print("LIGER KERNEL GRADIENT ACCUMULATION STABILITY TEST")
    print("="*80)
    
    # Configuration
    vocab_size = 10000
    hidden_dim = 512
    seq_len = 128
    batch_size = 4
    num_samples = 100
    gradient_accumulation_steps = 4
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Vocab size: {vocab_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Sequence length: {seq_len}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    
    # Create dataset and dataloader
    dataset = DummyTextDataset(vocab_size, seq_len, num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    results = {}
    
    # Test 1: Baseline (Standard PyTorch)
    print("\n" + "-"*60)
    print("TEST 1: Baseline (Standard PyTorch Cross Entropy)")
    print("-"*60)
    
    torch.manual_seed(42)
    model_baseline = SimpleTransformerModel(vocab_size, hidden_dim)
    results["baseline"] = train_baseline(
        model_baseline, dataloader, gradient_accumulation_steps=gradient_accumulation_steps, device=device
    )
    
    print(f"✓ Completed")
    print(f"  Avg loss: {results['baseline']['avg_loss']:.4f}")
    print(f"  Loss std: {results['baseline']['loss_std']:.4f}")
    print(f"  Time: {results['baseline']['total_time']:.2f}s")
    
    # Test 2: Liger Kernel with Gradient Accumulation
    print("\n" + "-"*60)
    print("TEST 2: Liger Kernel with Gradient Accumulation")
    print("-"*60)
    
    torch.manual_seed(42)
    model_liger = SimpleTransformerModel(vocab_size, hidden_dim)
    results["liger"] = train_with_liger(
        model_liger, dataloader, gradient_accumulation_steps=gradient_accumulation_steps, device=device
    )
    
    print(f"✓ Completed")
    print(f"  Avg loss: {results['liger']['avg_loss']:.4f}")
    print(f"  Loss std: {results['liger']['loss_std']:.4f}")
    print(f"  Time: {results['liger']['total_time']:.2f}s")
    
    # Test 3: Different accumulation steps
    print("\n" + "-"*60)
    print("TEST 3: Liger Kernel with Higher Accumulation (8 steps)")
    print("-"*60)
    
    torch.manual_seed(42)
    model_liger_8 = SimpleTransformerModel(vocab_size, hidden_dim)
    results["liger_8_steps"] = train_with_liger(
        model_liger_8, dataloader, gradient_accumulation_steps=8, device=device
    )
    
    print(f"✓ Completed")
    print(f"  Avg loss: {results['liger_8_steps']['avg_loss']:.4f}")
    print(f"  Loss std: {results['liger_8_steps']['loss_std']:.4f}")
    print(f"  Time: {results['liger_8_steps']['total_time']:.2f}s")
    
    # Stability Analysis
    print("\n" + "="*80)
    print("STABILITY ANALYSIS")
    print("="*80)
    
    baseline_metrics = results["baseline"]
    liger_metrics = results["liger"]
    liger_8_metrics = results["liger_8_steps"]
    
    print("\n1. Loss Stability")
    print("-" * 40)
    print(f"Baseline:        μ={baseline_metrics['avg_loss']:.4f}, σ={baseline_metrics['loss_std']:.4f}")
    print(f"Liger (4 steps): μ={liger_metrics['avg_loss']:.4f}, σ={liger_metrics['loss_std']:.4f}")
    print(f"Liger (8 steps): μ={liger_8_metrics['avg_loss']:.4f}, σ={liger_8_metrics['loss_std']:.4f}")
    
    loss_stability_4 = liger_metrics['loss_std'] / baseline_metrics['loss_std']
    loss_stability_8 = liger_8_metrics['loss_std'] / baseline_metrics['loss_std']
    
    print(f"\nStability ratio (σ_liger/σ_baseline):")
    print(f"  4 steps: {loss_stability_4:.3f} {'✓ Stable' if loss_stability_4 < 1.5 else '⚠ Higher variance'}")
    print(f"  8 steps: {loss_stability_8:.3f} {'✓ Stable' if loss_stability_8 < 1.5 else '⚠ Higher variance'}")
    
    print("\n2. Gradient Stability")
    print("-" * 40)
    print(f"Baseline:        μ={baseline_metrics['avg_grad_norm']:.2f}, σ={baseline_metrics['grad_norm_std']:.2f}")
    print(f"Liger (4 steps): μ={liger_metrics['avg_grad_norm']:.2f}, σ={liger_metrics['grad_norm_std']:.2f}")
    print(f"Liger (8 steps): μ={liger_8_metrics['avg_grad_norm']:.2f}, σ={liger_8_metrics['grad_norm_std']:.2f}")
    
    grad_stability_4 = liger_metrics['grad_norm_std'] / baseline_metrics['grad_norm_std']
    grad_stability_8 = liger_8_metrics['grad_norm_std'] / baseline_metrics['grad_norm_std']
    
    print(f"\nStability ratio:")
    print(f"  4 steps: {grad_stability_4:.3f} {'✓ Stable' if grad_stability_4 < 1.5 else '⚠ Higher variance'}")
    print(f"  8 steps: {grad_stability_8:.3f} {'✓ Stable' if grad_stability_8 < 1.5 else '⚠ Higher variance'}")
    
    print("\n3. Performance Comparison")
    print("-" * 40)
    
    speedup_4 = baseline_metrics['total_time'] / liger_metrics['total_time']
    speedup_8 = baseline_metrics['total_time'] / liger_8_metrics['total_time']
    
    print(f"Training time:")
    print(f"  Baseline:        {baseline_metrics['total_time']:.2f}s")
    print(f"  Liger (4 steps): {liger_metrics['total_time']:.2f}s (speedup: {speedup_4:.2f}x)")
    print(f"  Liger (8 steps): {liger_8_metrics['total_time']:.2f}s (speedup: {speedup_8:.2f}x)")
    
    if torch.cuda.is_available():
        print(f"\nPeak memory usage:")
        print(f"  Baseline:        {baseline_metrics['peak_memory_gb']:.3f} GB")
        print(f"  Liger (4 steps): {liger_metrics['peak_memory_gb']:.3f} GB")
        print(f"  Liger (8 steps): {liger_8_metrics['peak_memory_gb']:.3f} GB")
        
        mem_reduction_4 = (1 - liger_metrics['peak_memory_gb'] / baseline_metrics['peak_memory_gb']) * 100
        mem_reduction_8 = (1 - liger_8_metrics['peak_memory_gb'] / baseline_metrics['peak_memory_gb']) * 100
        
        print(f"\nMemory reduction:")
        print(f"  4 steps: {mem_reduction_4:.1f}% {'✓' if mem_reduction_4 > 0 else ''}")
        print(f"  8 steps: {mem_reduction_8:.1f}% {'✓' if mem_reduction_8 > 0 else ''}")
    
    # Overall verdict
    print("\n" + "="*80)
    print("OVERALL VERDICT")
    print("="*80)
    
    stability_ok = loss_stability_4 < 1.5 and grad_stability_4 < 1.5
    performance_ok = speedup_4 >= 0.9  # At least 90% of baseline speed
    
    if stability_ok and performance_ok:
        print("✓ Liger Kernel with gradient accumulation is STABLE and PERFORMANT")
        print("  - Loss variance is comparable to baseline")
        print("  - Gradient norms are stable")
        print("  - Performance is maintained or improved")
        if torch.cuda.is_available() and mem_reduction_4 > 0:
            print(f"  - Memory usage reduced by {mem_reduction_4:.1f}%")
    else:
        print("⚠ Issues detected:")
        if not stability_ok:
            print("  - Stability concerns (higher variance in loss or gradients)")
        if not performance_ok:
            print("  - Performance degradation detected")
    
    # Save results
    with open("stability_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print("\nDetailed results saved to stability_test_results.json")
    
    return results


if __name__ == "__main__":
    try:
        results = run_stability_comparison()
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)