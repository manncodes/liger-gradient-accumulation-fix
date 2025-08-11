#!/usr/bin/env python3
"""
Memory and performance profiling for Liger Kernel gradient accumulation.
This module provides comprehensive profiling tools to analyze memory usage,
execution time, and efficiency gains when using Liger Kernel.
"""

import torch
import torch.nn as nn
import time
import psutil
import os
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
from contextlib import contextmanager
import sys
import tracemalloc

# Add Liger Kernel to path if available
try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    print("Warning: Liger Kernel not available. Install with: pip install liger-kernel")


class MemoryProfiler:
    """Context manager for detailed memory profiling."""
    
    def __init__(self, description: str = ""):
        self.description = description
        self.start_memory = 0
        self.peak_memory = 0
        self.end_memory = 0
        self.cuda_start = 0
        self.cuda_peak = 0
        self.cuda_end = 0
        
    def __enter__(self):
        # CPU memory
        process = psutil.Process(os.getpid())
        self.start_memory = process.memory_info().rss / (1024**3)  # GB
        
        # GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.cuda_start = torch.cuda.memory_allocated() / (1024**3)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # CPU memory
        process = psutil.Process(os.getpid())
        self.end_memory = process.memory_info().rss / (1024**3)
        
        # GPU memory
        if torch.cuda.is_available():
            self.cuda_peak = torch.cuda.max_memory_allocated() / (1024**3)
            self.cuda_end = torch.cuda.memory_allocated() / (1024**3)
        
        print(f"Memory Profile - {self.description}:")
        print(f"  CPU: {self.start_memory:.3f}GB -> {self.end_memory:.3f}GB (delta: {self.end_memory - self.start_memory:+.3f}GB)")
        if torch.cuda.is_available():
            print(f"  GPU: {self.cuda_start:.3f}GB -> {self.cuda_end:.3f}GB (peak: {self.cuda_peak:.3f}GB)")


@contextmanager
def time_profiler(description: str = ""):
    """Context manager for timing operations."""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    print(f"Time Profile - {description}: {end_time - start_time:.4f}s")


def profile_memory_usage(
    vocab_size: int = 50000,
    hidden_dim: int = 4096,
    batch_size: int = 8,
    seq_len: int = 1024,
    accumulation_steps: int = 4,
    device: str = "cuda"
) -> Dict:
    """
    Profile memory usage comparing PyTorch baseline vs Liger Kernel.
    
    Args:
        vocab_size: Size of vocabulary
        hidden_dim: Hidden dimension size
        batch_size: Batch size per step
        seq_len: Sequence length
        accumulation_steps: Number of gradient accumulation steps
        device: Device to run on
    
    Returns:
        Dictionary containing profiling results
    """
    
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("Warning: CUDA not available, using CPU")
    
    if not LIGER_AVAILABLE:
        print("Error: Liger Kernel not available for profiling")
        return {}
    
    print(f"\n=== MEMORY PROFILING ===")
    print(f"Configuration:")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Hidden dim: {hidden_dim:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Accumulation steps: {accumulation_steps}")
    print(f"  Effective batch size: {batch_size * accumulation_steps}")
    print(f"  Device: {device}")
    
    # Create model components
    linear = nn.Linear(hidden_dim, vocab_size, bias=True).to(device)
    
    # Generate test data
    torch.manual_seed(42)
    inputs = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    results = {
        "config": {
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "accumulation_steps": accumulation_steps,
            "device": device
        }
    }
    
    # Profile PyTorch baseline
    print(f"\n--- PyTorch Baseline ---")
    with MemoryProfiler("PyTorch Baseline") as mem_baseline:
        with time_profiler("PyTorch Forward+Backward"):
            # Standard approach - materializes full logits tensor
            criterion = nn.CrossEntropyLoss(reduction='mean')
            
            total_loss = 0
            for step in range(accumulation_steps):
                # Forward pass
                logits = linear(inputs)  # This creates the large logits tensor
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                
                loss = criterion(logits_flat, targets_flat)
                loss = loss / accumulation_steps
                loss.backward()
                
                total_loss += loss.item() * accumulation_steps
    
    results["pytorch"] = {
        "cpu_memory_delta": mem_baseline.end_memory - mem_baseline.start_memory,
        "gpu_memory_peak": mem_baseline.cuda_peak if torch.cuda.is_available() else 0,
        "total_loss": total_loss
    }
    
    # Clean up
    linear.zero_grad()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Profile Liger Kernel
    print(f"\n--- Liger Kernel ---")
    with MemoryProfiler("Liger Kernel") as mem_liger:
        with time_profiler("Liger Forward+Backward"):
            # Liger approach - no logits materialization
            liger_criterion = LigerFusedLinearCrossEntropyLoss(reduction='mean')
            
            total_loss = 0
            for step in range(accumulation_steps):
                # Forward pass with Liger
                inputs_flat = inputs.view(-1, hidden_dim)
                targets_flat = targets.view(-1)
                
                loss = liger_criterion(
                    linear.weight,
                    inputs_flat,
                    targets_flat,
                    linear.bias
                )
                loss = loss / accumulation_steps
                loss.backward()
                
                total_loss += loss.item() * accumulation_steps
    
    results["liger"] = {
        "cpu_memory_delta": mem_liger.end_memory - mem_liger.start_memory,
        "gpu_memory_peak": mem_liger.cuda_peak if torch.cuda.is_available() else 0,
        "total_loss": total_loss
    }
    
    # Calculate savings
    if torch.cuda.is_available():
        memory_savings = results["pytorch"]["gpu_memory_peak"] - results["liger"]["gpu_memory_peak"]
        savings_percent = (memory_savings / results["pytorch"]["gpu_memory_peak"]) * 100 if results["pytorch"]["gpu_memory_peak"] > 0 else 0
        
        print(f"\n--- Memory Savings Analysis ---")
        print(f"PyTorch peak GPU memory: {results['pytorch']['gpu_memory_peak']:.3f} GB")
        print(f"Liger peak GPU memory: {results['liger']['gpu_memory_peak']:.3f} GB")
        print(f"Memory savings: {memory_savings:.3f} GB ({savings_percent:.1f}%)")
        
        results["savings"] = {
            "absolute_gb": memory_savings,
            "percent": savings_percent
        }
    
    # Verify mathematical correctness
    loss_diff = abs(results["pytorch"]["total_loss"] - results["liger"]["total_loss"])
    print(f"\nLoss verification:")
    print(f"PyTorch loss: {results['pytorch']['total_loss']:.8f}")
    print(f"Liger loss: {results['liger']['total_loss']:.8f}")
    print(f"Difference: {loss_diff:.8f} ({'PASS' if loss_diff < 1e-5 else 'FAIL'})")
    
    results["correctness"] = {
        "loss_difference": loss_diff,
        "mathematically_correct": loss_diff < 1e-5
    }
    
    return results


def profile_gradient_accumulation_scaling(
    base_vocab_size: int = 10000,
    hidden_dim: int = 2048,
    batch_size: int = 4,
    seq_len: int = 256,
    max_accumulation_steps: int = 16,
    device: str = "cuda"
) -> Dict:
    """
    Profile how memory usage scales with gradient accumulation steps.
    
    Returns:
        Dictionary with scaling analysis results
    """
    
    if not LIGER_AVAILABLE:
        return {}
    
    print(f"\n=== GRADIENT ACCUMULATION SCALING ANALYSIS ===")
    
    results = {"scaling_data": []}
    
    accumulation_steps_list = [1, 2, 4, 8, 16]
    if max_accumulation_steps < 16:
        accumulation_steps_list = [s for s in accumulation_steps_list if s <= max_accumulation_steps]
    
    for steps in accumulation_steps_list:
        print(f"\nTesting {steps} accumulation steps...")
        
        # Test with current step count
        step_results = profile_memory_usage(
            vocab_size=base_vocab_size,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            seq_len=seq_len,
            accumulation_steps=steps,
            device=device
        )
        
        if step_results:
            step_data = {
                "accumulation_steps": steps,
                "effective_batch_size": batch_size * steps,
                "pytorch_memory": step_results.get("pytorch", {}).get("gpu_memory_peak", 0),
                "liger_memory": step_results.get("liger", {}).get("gpu_memory_peak", 0),
                "memory_savings_percent": step_results.get("savings", {}).get("percent", 0)
            }
            results["scaling_data"].append(step_data)
    
    # Analyze scaling trends
    if results["scaling_data"]:
        print(f"\n--- Scaling Analysis Summary ---")
        print(f"{'Steps':<6} {'Eff.Batch':<10} {'PyTorch(GB)':<12} {'Liger(GB)':<10} {'Savings(%)':<10}")
        print(f"{'-'*6} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
        
        for data in results["scaling_data"]:
            print(f"{data['accumulation_steps']:<6} "
                  f"{data['effective_batch_size']:<10} "
                  f"{data['pytorch_memory']:<12.3f} "
                  f"{data['liger_memory']:<10.3f} "
                  f"{data['memory_savings_percent']:<10.1f}")
    
    return results


def profile_vocabulary_scaling(
    vocab_sizes: List[int] = None,
    hidden_dim: int = 2048,
    batch_size: int = 4,
    seq_len: int = 256,
    accumulation_steps: int = 4,
    device: str = "cuda"
) -> Dict:
    """
    Profile how memory usage scales with vocabulary size.
    
    Returns:
        Dictionary with vocabulary scaling results
    """
    
    if vocab_sizes is None:
        vocab_sizes = [1000, 5000, 10000, 25000, 50000]
    
    if not LIGER_AVAILABLE:
        return {}
    
    print(f"\n=== VOCABULARY SIZE SCALING ANALYSIS ===")
    
    results = {"vocab_scaling_data": []}
    
    for vocab_size in vocab_sizes:
        print(f"\nTesting vocabulary size: {vocab_size:,}")
        
        try:
            vocab_results = profile_memory_usage(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                batch_size=batch_size,
                seq_len=seq_len,
                accumulation_steps=accumulation_steps,
                device=device
            )
            
            if vocab_results:
                vocab_data = {
                    "vocab_size": vocab_size,
                    "pytorch_memory": vocab_results.get("pytorch", {}).get("gpu_memory_peak", 0),
                    "liger_memory": vocab_results.get("liger", {}).get("gpu_memory_peak", 0),
                    "memory_savings_gb": vocab_results.get("savings", {}).get("absolute_gb", 0),
                    "memory_savings_percent": vocab_results.get("savings", {}).get("percent", 0),
                    "mathematically_correct": vocab_results.get("correctness", {}).get("mathematically_correct", False)
                }
                results["vocab_scaling_data"].append(vocab_data)
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Skipping vocab size {vocab_size:,} - GPU OOM")
                continue
            else:
                raise
    
    # Summary
    if results["vocab_scaling_data"]:
        print(f"\n--- Vocabulary Scaling Summary ---")
        print(f"{'Vocab Size':<12} {'PyTorch(GB)':<12} {'Liger(GB)':<10} {'Savings(GB)':<12} {'Savings(%)':<10} {'Correct':<8}")
        print(f"{'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")
        
        for data in results["vocab_scaling_data"]:
            print(f"{data['vocab_size']:<12,} "
                  f"{data['pytorch_memory']:<12.3f} "
                  f"{data['liger_memory']:<10.3f} "
                  f"{data['memory_savings_gb']:<12.3f} "
                  f"{data['memory_savings_percent']:<10.1f} "
                  f"{'YES' if data['mathematically_correct'] else 'NO':<8}")
    
    return results


def comprehensive_profiling_suite(output_file: str = "profiling_results.json"):
    """
    Run comprehensive profiling suite and save results to JSON.
    
    Args:
        output_file: Path to save profiling results
    """
    
    print("COMPREHENSIVE LIGER KERNEL PROFILING SUITE")
    print("=" * 60)
    
    if not LIGER_AVAILABLE:
        print("ERROR: Liger Kernel not available. Cannot run profiling.")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    all_results = {
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "liger_available": LIGER_AVAILABLE,
        "torch_version": torch.__version__
    }
    
    try:
        # 1. Basic memory profiling
        print(f"\n1. BASIC MEMORY PROFILING")
        basic_results = profile_memory_usage(
            vocab_size=25000,
            hidden_dim=2048,
            batch_size=4,
            seq_len=512,
            accumulation_steps=4,
            device=device
        )
        all_results["basic_profiling"] = basic_results
        
        # 2. Gradient accumulation scaling
        print(f"\n2. GRADIENT ACCUMULATION SCALING")
        scaling_results = profile_gradient_accumulation_scaling(
            base_vocab_size=15000,
            hidden_dim=1024,
            batch_size=2,
            seq_len=256,
            max_accumulation_steps=8,
            device=device
        )
        all_results["accumulation_scaling"] = scaling_results
        
        # 3. Vocabulary scaling  
        print(f"\n3. VOCABULARY SIZE SCALING")
        vocab_results = profile_vocabulary_scaling(
            vocab_sizes=[5000, 15000, 30000] if device == "cuda" else [1000, 5000, 10000],
            hidden_dim=1024,
            batch_size=2,
            seq_len=256,
            accumulation_steps=2,
            device=device
        )
        all_results["vocabulary_scaling"] = vocab_results
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        all_results["error"] = str(e)
    
    # Save results
    try:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"\nProfiling results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # Summary
    print(f"\n" + "="*60)
    print("PROFILING SUMMARY")
    print("="*60)
    
    if "basic_profiling" in all_results and all_results["basic_profiling"]:
        basic = all_results["basic_profiling"]
        if "savings" in basic:
            savings = basic["savings"]
            print(f"Memory savings: {savings['absolute_gb']:.2f} GB ({savings['percent']:.1f}%)")
        
        if "correctness" in basic:
            correct = basic["correctness"]["mathematically_correct"]
            print(f"Mathematical correctness: {'VERIFIED' if correct else 'FAILED'}")
    
    print(f"Device used: {device}")
    print(f"Results saved: {output_file}")
    print("Profiling complete!")


if __name__ == "__main__":
    # Run comprehensive profiling
    comprehensive_profiling_suite()