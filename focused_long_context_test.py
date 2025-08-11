#!/usr/bin/env python3
"""
Focused long context memory test with realistic parameters.
"""

import torch
import torch.nn as nn
import time
import json
import sys
import os

# Add Liger Kernel to path if available
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Liger-Kernel', 'src'))

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    print("Error: Liger Kernel not available")
    sys.exit(1)


def test_long_context_scenario(seq_len: int, vocab_size: int = 50000, hidden_dim: int = 4096):
    """Test a specific long context scenario."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*80}")
    print(f"TESTING {seq_len//1024}K CONTEXT LENGTH")
    print(f"{'='*80}")
    print(f"Sequence length: {seq_len:,} tokens")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Hidden dimension: {hidden_dim:,}")
    print(f"Batch size: 1 (typical for long context)")
    
    # Calculate expected memory usage
    logits_size_gb = (seq_len * vocab_size * 4) / (1024**3)  # 4 bytes per float32
    print(f"Expected logits tensor size: {logits_size_gb:.2f} GB")
    
    # Create model and data
    linear = nn.Linear(hidden_dim, vocab_size, bias=True).to(device)
    
    # Use smaller dtype for input to save memory during generation
    print("Generating test data...")
    torch.manual_seed(42)
    
    try:
        # Generate in chunks to avoid OOM
        chunk_size = 8192
        num_chunks = seq_len // chunk_size
        remaining = seq_len % chunk_size
        
        input_chunks = []
        target_chunks = []
        
        for i in range(num_chunks):
            input_chunk = torch.randn(1, chunk_size, hidden_dim, device=device, dtype=torch.float16)
            target_chunk = torch.randint(0, vocab_size, (1, chunk_size), device=device)
            input_chunks.append(input_chunk)
            target_chunks.append(target_chunk)
        
        if remaining > 0:
            input_chunk = torch.randn(1, remaining, hidden_dim, device=device, dtype=torch.float16)
            target_chunk = torch.randint(0, vocab_size, (1, remaining), device=device)
            input_chunks.append(input_chunk)
            target_chunks.append(target_chunk)
        
        # Concatenate
        inputs = torch.cat(input_chunks, dim=1).float()
        targets = torch.cat(target_chunks, dim=1)
        
        print(f"Test data generated: {inputs.shape}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Cannot generate test data - OOM")
            return {"error": "Data generation OOM", "seq_len": seq_len}
        else:
            raise
    
    results = {}
    
    # Test 1: PyTorch Baseline
    print(f"\n{'-'*40}")
    print("PyTorch Baseline Test")
    print(f"{'-'*40}")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated() / (1024**3)
    
    try:
        start_time = time.time()
        
        criterion = nn.CrossEntropyLoss(reduction='mean')
        
        # This will create the massive logits tensor
        logits = linear(inputs)  # [1, seq_len, vocab_size]
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        
        end_time = time.time()
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        
        results["pytorch"] = {
            "success": True,
            "loss": loss.item(),
            "time": end_time - start_time,
            "start_memory_gb": start_mem,
            "peak_memory_gb": peak_mem,
            "memory_used_gb": peak_mem - start_mem
        }
        
        print(f"SUCCESS: Loss = {loss.item():.6f}")
        print(f"Time: {end_time - start_time:.2f}s")
        print(f"Peak memory: {peak_mem:.2f} GB")
        print(f"Memory used: {peak_mem - start_mem:.2f} GB")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            results["pytorch"] = {"success": False, "error": "OOM"}
            print(f"FAILED: Out of Memory")
        else:
            raise
    
    # Clear memory
    linear.zero_grad()
    torch.cuda.empty_cache()
    
    # Test 2: Liger Kernel
    print(f"\n{'-'*40}")
    print("Liger Kernel Test")
    print(f"{'-'*40}")
    
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated() / (1024**3)
    
    try:
        start_time = time.time()
        
        liger_criterion = LigerFusedLinearCrossEntropyLoss(reduction='mean')
        
        # Liger approach - no logits materialization
        inputs_flat = inputs.view(-1, hidden_dim)
        targets_flat = targets.view(-1)
        
        loss = liger_criterion(
            linear.weight,
            inputs_flat,
            targets_flat,
            linear.bias
        )
        loss.backward()
        
        end_time = time.time()
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        
        results["liger"] = {
            "success": True,
            "loss": loss.item(),
            "time": end_time - start_time,
            "start_memory_gb": start_mem,
            "peak_memory_gb": peak_mem,
            "memory_used_gb": peak_mem - start_mem
        }
        
        print(f"SUCCESS: Loss = {loss.item():.6f}")
        print(f"Time: {end_time - start_time:.2f}s")
        print(f"Peak memory: {peak_mem:.2f} GB")
        print(f"Memory used: {peak_mem - start_mem:.2f} GB")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            results["liger"] = {"success": False, "error": "OOM"}
            print(f"FAILED: Out of Memory")
        else:
            raise
    
    # Analysis
    print(f"\n{'-'*40}")
    print("Analysis")
    print(f"{'-'*40}")
    
    if results.get("pytorch", {}).get("success") and results.get("liger", {}).get("success"):
        pytorch_mem = results["pytorch"]["peak_memory_gb"]
        liger_mem = results["liger"]["peak_memory_gb"]
        
        memory_savings = pytorch_mem - liger_mem
        savings_percent = (memory_savings / pytorch_mem) * 100
        
        pytorch_time = results["pytorch"]["time"]
        liger_time = results["liger"]["time"]
        speedup = pytorch_time / liger_time
        
        loss_diff = abs(results["pytorch"]["loss"] - results["liger"]["loss"])
        
        print(f"Memory Comparison:")
        print(f"  PyTorch peak: {pytorch_mem:.2f} GB")
        print(f"  Liger peak:   {liger_mem:.2f} GB")
        print(f"  Savings:      {memory_savings:.2f} GB ({savings_percent:.1f}%)")
        print(f"")
        print(f"Performance:")
        print(f"  PyTorch time: {pytorch_time:.2f}s")
        print(f"  Liger time:   {liger_time:.2f}s")
        print(f"  Speedup:      {speedup:.2f}x")
        print(f"")
        print(f"Correctness:")
        print(f"  Loss difference: {loss_diff:.8f} ({'PASS' if loss_diff < 1e-4 else 'FAIL'})")
        
        results["analysis"] = {
            "memory_savings_gb": memory_savings,
            "memory_savings_percent": savings_percent,
            "speedup": speedup,
            "loss_difference": loss_diff,
            "mathematically_correct": loss_diff < 1e-4,
            "expected_logits_size_gb": logits_size_gb,
            "actual_savings_vs_expected": (memory_savings / logits_size_gb) * 100 if logits_size_gb > 0 else 0
        }
        
    elif results.get("pytorch", {}).get("success") == False and results.get("liger", {}).get("success"):
        liger_mem = results["liger"]["peak_memory_gb"]
        print(f"BREAKTHROUGH: PyTorch failed (OOM), Liger succeeded!")
        print(f"Liger memory usage: {liger_mem:.2f} GB")
        print(f"This scenario is ONLY possible with Liger Kernel")
        
        results["analysis"] = {
            "pytorch_oom": True,
            "liger_success": True,
            "liger_memory_gb": liger_mem,
            "breakthrough": True,
            "expected_logits_size_gb": logits_size_gb
        }
        
    elif results.get("pytorch", {}).get("success") and results.get("liger", {}).get("success") == False:
        print(f"Unexpected: Liger failed but PyTorch succeeded")
        print(f"This suggests a bug in our implementation")
        
    else:
        print(f"Both approaches failed - sequence too long for available hardware")
    
    return results


def run_long_context_tests():
    """Run tests for different long context scenarios."""
    
    print("LONG CONTEXT SFT MEMORY PROFILING")
    print("Demonstrating Liger Kernel benefits for 64K+ sequences")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA required for long context testing")
        return
    
    # Show GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    # Test scenarios - realistic long context SFT scenarios
    scenarios = [
        {"name": "32K Context", "seq_len": 32768, "vocab_size": 32000},
        {"name": "64K Context", "seq_len": 65536, "vocab_size": 32000},
        {"name": "128K Context", "seq_len": 131072, "vocab_size": 50000},
    ]
    
    all_results = {
        "gpu_info": {"name": gpu_name, "memory_gb": gpu_memory},
        "scenarios": {}
    }
    
    for scenario in scenarios:
        scenario_name = scenario["name"]
        seq_len = scenario["seq_len"]
        vocab_size = scenario["vocab_size"]
        
        print(f"\n" + "="*80)
        print(f"SCENARIO: {scenario_name}")
        
        try:
            result = test_long_context_scenario(seq_len, vocab_size)
            all_results["scenarios"][scenario_name] = result
            
            # Don't continue if both approaches fail
            if (not result.get("pytorch", {}).get("success", False) and 
                not result.get("liger", {}).get("success", False)):
                print(f"Both approaches failed - hardware limit reached")
                break
                
        except Exception as e:
            print(f"Error in {scenario_name}: {e}")
            all_results["scenarios"][scenario_name] = {"error": str(e)}
            break
    
    # Save results
    output_file = "long_context_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    
    # Final summary
    print(f"\n" + "="*80)
    print("LONG CONTEXT SFT SUMMARY")
    print("="*80)
    
    print(f"{'Scenario':<15} {'PyTorch':<12} {'Liger':<12} {'Savings':<12} {'Breakthrough'}")
    print(f"{'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for scenario_name, result in all_results["scenarios"].items():
        if "error" in result:
            print(f"{scenario_name:<15} {'ERROR':<12} {'ERROR':<12} {'N/A':<12} {'N/A'}")
            continue
            
        pytorch_ok = result.get("pytorch", {}).get("success", False)
        liger_ok = result.get("liger", {}).get("success", False)
        
        pytorch_status = "SUCCESS" if pytorch_ok else "OOM"
        liger_status = "SUCCESS" if liger_ok else "OOM"
        
        if "analysis" in result:
            analysis = result["analysis"]
            if "memory_savings_percent" in analysis:
                savings = f"{analysis['memory_savings_percent']:.1f}%"
            else:
                savings = "N/A"
                
            breakthrough = "YES" if analysis.get("breakthrough", False) else "NO"
        else:
            savings = "N/A"
            breakthrough = "N/A"
        
        print(f"{scenario_name:<15} {pytorch_status:<12} {liger_status:<12} {savings:<12} {breakthrough}")
    
    print(f"\nKey Insights:")
    print(f"- Memory savings increase dramatically with longer contexts")
    print(f"- Some scenarios may only be trainable with Liger Kernel")
    print(f"- Long context SFT benefits most from Liger's memory optimization")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    run_long_context_tests()