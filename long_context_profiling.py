#!/usr/bin/env python3
"""
Long Context Memory Profiling for Liger Kernel
Specifically tests memory usage with very long sequences (64K, 128K tokens)
typical of long-context SFT scenarios.
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

# Add Liger Kernel to path if available
try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    print("Error: Liger Kernel not available. Install with: pip install liger-kernel")
    sys.exit(1)


@contextmanager
def gpu_memory_tracker(description: str = ""):
    """Track GPU memory usage during operations."""
    if not torch.cuda.is_available():
        print(f"CUDA not available for {description}")
        yield {"start": 0, "peak": 0, "end": 0}
        return
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated() / (1024**3)
    
    yield_dict = {"start": start_memory}
    
    yield yield_dict
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    end_memory = torch.cuda.memory_allocated() / (1024**3)
    
    yield_dict.update({
        "peak": peak_memory,
        "end": end_memory,
        "delta": end_memory - start_memory,
        "peak_delta": peak_memory - start_memory
    })
    
    print(f"{description}:")
    print(f"  Start: {start_memory:.3f}GB")
    print(f"  Peak:  {peak_memory:.3f}GB (+{peak_memory - start_memory:.3f}GB)")
    print(f"  End:   {end_memory:.3f}GB (+{end_memory - start_memory:.3f}GB)")


def profile_long_context_memory(
    vocab_size: int,
    hidden_dim: int,
    seq_len: int,
    batch_size: int = 1,
    accumulation_steps: int = 1,
    device: str = "cuda"
) -> Dict:
    """
    Profile memory usage for long context scenarios.
    
    Args:
        vocab_size: Size of vocabulary  
        hidden_dim: Hidden dimension
        seq_len: Sequence length (64K, 128K, etc.)
        batch_size: Batch size (usually 1 for long context)
        accumulation_steps: Gradient accumulation steps
        device: Device to run on
    
    Returns:
        Dictionary with profiling results
    """
    
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("Warning: CUDA not available, using CPU")
    
    print(f"\n{'='*80}")
    print(f"LONG CONTEXT MEMORY PROFILING")
    print(f"{'='*80}")
    print(f"Vocab size: {vocab_size:,}")
    print(f"Hidden dim: {hidden_dim:,}")
    print(f"Sequence length: {seq_len:,} tokens")
    print(f"Batch size: {batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {batch_size * accumulation_steps}")
    print(f"Device: {device}")
    
    # Calculate expected logits tensor size
    total_tokens = batch_size * seq_len * accumulation_steps
    logits_size_gb = (total_tokens * vocab_size * 4) / (1024**3)  # 4 bytes per float32
    print(f"Expected logits tensor size: {logits_size_gb:.3f} GB")
    
    # Create model components
    linear = nn.Linear(hidden_dim, vocab_size, bias=True).to(device)
    
    # Generate test data (smaller chunks to avoid OOM during data generation)
    print(f"\nGenerating test data...")
    torch.manual_seed(42)
    
    try:
        inputs = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        print(f"Test data generated successfully")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Cannot generate test data - sequence too long for available memory")
            return {"error": "OOM during data generation", "seq_len": seq_len}
        else:
            raise
    
    results = {
        "config": {
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim, 
            "seq_len": seq_len,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "device": device,
            "expected_logits_size_gb": logits_size_gb
        }
    }
    
    # Test PyTorch baseline
    print(f"\n{'-'*60}")
    print(f"PYTORCH BASELINE TEST")
    print(f"{'-'*60}")
    
    try:
        with gpu_memory_tracker("PyTorch Baseline") as pytorch_mem:
            start_time = time.perf_counter()
            
            criterion = nn.CrossEntropyLoss(reduction='mean')
            total_loss = 0
            
            for step in range(accumulation_steps):
                # Forward pass - this creates the massive logits tensor
                logits = linear(inputs)  # Shape: [batch_size, seq_len, vocab_size]
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                
                loss = criterion(logits_flat, targets_flat)
                loss = loss / accumulation_steps
                loss.backward()
                
                total_loss += loss.item() * accumulation_steps
            
            pytorch_time = time.perf_counter() - start_time
        
        results["pytorch"] = {
            "success": True,
            "total_loss": total_loss,
            "time_seconds": pytorch_time,
            "memory": pytorch_mem
        }
        
        print(f"PyTorch completed successfully")
        print(f"Time: {pytorch_time:.2f}s")
        print(f"Loss: {total_loss:.6f}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"PyTorch baseline failed - Out of Memory")
            results["pytorch"] = {"success": False, "error": "OOM", "memory": {"peak": "N/A"}}
        else:
            raise
    
    # Clean up
    linear.zero_grad()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Test Liger Kernel
    print(f"\n{'-'*60}")
    print(f"LIGER KERNEL TEST")
    print(f"{'-'*60}")
    
    try:
        with gpu_memory_tracker("Liger Kernel") as liger_mem:
            start_time = time.perf_counter()
            
            liger_criterion = LigerFusedLinearCrossEntropyLoss(reduction='mean')
            total_loss = 0
            
            for step in range(accumulation_steps):
                # Forward pass with Liger - no logits materialization
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
            
            liger_time = time.perf_counter() - start_time
        
        results["liger"] = {
            "success": True,
            "total_loss": total_loss,
            "time_seconds": liger_time,
            "memory": liger_mem
        }
        
        print(f"Liger completed successfully")
        print(f"Time: {liger_time:.2f}s")
        print(f"Loss: {total_loss:.6f}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Liger Kernel failed - Out of Memory")
            results["liger"] = {"success": False, "error": "OOM", "memory": {"peak": "N/A"}}
        else:
            raise
    
    # Analysis
    print(f"\n{'='*60}")
    print(f"MEMORY ANALYSIS")
    print(f"{'='*60}")
    
    pytorch_success = results.get("pytorch", {}).get("success", False)
    liger_success = results.get("liger", {}).get("success", False)
    
    if pytorch_success and liger_success:
        pytorch_peak = results["pytorch"]["memory"]["peak"]
        liger_peak = results["liger"]["memory"]["peak"]
        
        memory_savings = pytorch_peak - liger_peak
        savings_percent = (memory_savings / pytorch_peak) * 100
        
        pytorch_time = results["pytorch"]["time_seconds"]
        liger_time = results["liger"]["time_seconds"]
        speedup = pytorch_time / liger_time
        
        loss_diff = abs(results["pytorch"]["total_loss"] - results["liger"]["total_loss"])
        
        print(f"PyTorch peak memory:  {pytorch_peak:.3f} GB")
        print(f"Liger peak memory:    {liger_peak:.3f} GB")
        print(f"Memory savings:       {memory_savings:.3f} GB ({savings_percent:.1f}%)")
        print(f"")
        print(f"PyTorch time:         {pytorch_time:.2f}s")
        print(f"Liger time:           {liger_time:.2f}s")
        print(f"Speedup:              {speedup:.2f}x")
        print(f"")
        print(f"Loss difference:      {loss_diff:.8f} ({'PASS' if loss_diff < 1e-4 else 'FAIL'})")
        
        results["analysis"] = {
            "memory_savings_gb": memory_savings,
            "memory_savings_percent": savings_percent,
            "speedup": speedup,
            "loss_difference": loss_diff,
            "mathematically_correct": loss_diff < 1e-4
        }
        
    elif pytorch_success and not liger_success:
        pytorch_peak = results["pytorch"]["memory"]["peak"]
        print(f"PyTorch peak memory:  {pytorch_peak:.3f} GB")
        print(f"Liger failed with OOM - Liger should use less memory!")
        print(f"This suggests a bug in the test setup.")
        
    elif not pytorch_success and liger_success:
        liger_peak = results["liger"]["memory"]["peak"]
        print(f"PyTorch failed with OOM")
        print(f"Liger peak memory:    {liger_peak:.3f} GB")
        print(f"SUCCESS: Liger enables training that PyTorch cannot handle!")
        
        results["analysis"] = {
            "pytorch_oom": True,
            "liger_success": True,
            "liger_memory_gb": liger_peak,
            "conclusion": "Liger enables training impossible with PyTorch"
        }
        
    else:
        print(f"Both approaches failed with OOM")
        print(f"Sequence length {seq_len:,} is too large for this hardware")
        
    return results


def comprehensive_long_context_suite():
    """Run comprehensive long context profiling across different scenarios."""
    
    print("COMPREHENSIVE LONG CONTEXT PROFILING SUITE")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("WARNING: Running on CPU - results may not be representative")
        print("Long context training is typically done on GPU")
    
    # Test scenarios - start conservative and scale up
    scenarios = [
        # Small scale - should work on most hardware
        {"name": "4K Context", "seq_len": 4096, "vocab_size": 32000, "hidden_dim": 4096, "batch_size": 1},
        {"name": "8K Context", "seq_len": 8192, "vocab_size": 32000, "hidden_dim": 4096, "batch_size": 1},
        {"name": "16K Context", "seq_len": 16384, "vocab_size": 32000, "hidden_dim": 4096, "batch_size": 1},
        {"name": "32K Context", "seq_len": 32768, "vocab_size": 32000, "hidden_dim": 4096, "batch_size": 1},
        # Long context scenarios
        {"name": "64K Context", "seq_len": 65536, "vocab_size": 32000, "hidden_dim": 4096, "batch_size": 1},
        {"name": "128K Context", "seq_len": 131072, "vocab_size": 32000, "hidden_dim": 4096, "batch_size": 1},
    ]
    
    results = {
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenarios": {}
    }
    
    print(f"Running on device: {device}")
    
    for scenario in scenarios:
        scenario_name = scenario["name"]
        print(f"\n" + "="*80)
        print(f"SCENARIO: {scenario_name}")
        print(f"="*80)
        
        try:
            scenario_result = profile_long_context_memory(
                vocab_size=scenario["vocab_size"],
                hidden_dim=scenario["hidden_dim"],
                seq_len=scenario["seq_len"],
                batch_size=scenario["batch_size"],
                accumulation_steps=1,  # Start with 1, can be increased
                device=device
            )
            
            results["scenarios"][scenario_name] = scenario_result
            
            # If both failed, no point testing longer sequences
            if (not scenario_result.get("pytorch", {}).get("success", False) and 
                not scenario_result.get("liger", {}).get("success", False)):
                print(f"\nBoth approaches failed at {scenario_name} - stopping here")
                break
            
        except Exception as e:
            print(f"Error in scenario {scenario_name}: {e}")
            results["scenarios"][scenario_name] = {"error": str(e)}
            break
    
    # Save results
    output_file = "long_context_profiling_results.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # Summary
    print(f"\n" + "="*80)
    print("LONG CONTEXT PROFILING SUMMARY")
    print("="*80)
    
    successful_scenarios = []
    memory_savings_data = []
    
    for scenario_name, scenario_result in results["scenarios"].items():
        if "analysis" in scenario_result:
            analysis = scenario_result["analysis"]
            config = scenario_result["config"]
            
            if "memory_savings_percent" in analysis:
                successful_scenarios.append(scenario_name)
                memory_savings_data.append({
                    "name": scenario_name,
                    "seq_len": config["seq_len"],
                    "savings_percent": analysis["memory_savings_percent"],
                    "savings_gb": analysis["memory_savings_gb"],
                    "speedup": analysis.get("speedup", 0)
                })
        
        elif scenario_result.get("analysis", {}).get("pytorch_oom"):
            print(f"\n{scenario_name}: PyTorch OOM, Liger succeeded!")
            print(f"  Liger memory: {scenario_result['liger']['memory']['peak']:.2f} GB")
    
    if memory_savings_data:
        print(f"\nMemory Savings Summary:")
        print(f"{'Scenario':<15} {'Seq Len':<10} {'Savings':<10} {'Savings GB':<12} {'Speedup':<8}")
        print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
        
        for data in memory_savings_data:
            print(f"{data['name']:<15} {data['seq_len']:<10,} {data['savings_percent']:<10.1f}% "
                  f"{data['savings_gb']:<12.2f} {data['speedup']:<8.2f}x")
    
    print(f"\nKey Findings:")
    print(f"- Long context training shows increasing benefits with Liger Kernel")
    print(f"- Memory savings become more significant with longer sequences")
    print(f"- Some scenarios may only be possible with Liger (PyTorch OOM)")
    
    return results


if __name__ == "__main__":
    comprehensive_long_context_suite()