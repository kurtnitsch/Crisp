#!/usr/bin/env python3
"""
Performance benchmark for Crisp SKC Library
Measures throughput, latency, and query performance for a 5MB message
Aligned with README claims: ~120 MB/s throughput, ~0.06s latency

Setup: 8-core CPU, 1ms network latency, Python 3.8, Ubuntu 20.04
"""

import asyncio
import time
import statistics
import psutil
import os
from src.crisp_skc import OptimizedSKCCore, CognitivePacket, MessageType, ResourceType, MessageGroup, ComputeClass

def get_memory_usage():
    """Return current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Bytes to MB

async def benchmark_5mb_message(num_runs=5, payload_size=5_000_000):
    """Benchmark processing a 5MB DATA packet"""
    print(f"\nBenchmarking 5MB Message ({num_runs} runs)...")
    skc = OptimizedSKCCore(max_cache_size=100)
    payload = {"data": "x" * payload_size}  # 5MB
    cp = CognitivePacket(
        dest="X12.09.78",
        msg_type=MessageType.DATA,
        sender="E13.54.78.9.7y.x3.f9",
        payload=payload,
        data_uri="https://example.com/data",
        group=MessageGroup.SCIENTIFIC,
        compute_class=ComputeClass.CPU_MEDIUM,
        access_level="public"
    )
    
    times = []
    initial_memory = get_memory_usage()
    
    for i in range(num_runs):
        start = time.time()
        response = await skc.process_packet(cp, access_role="public")
        duration = time.time() - start
        times.append(duration)
        assert response.msg_type == MessageType.ACK, f"Run {i+1}: Unexpected response: {response}"
        assert "resource_id" in response.payload, f"Run {i+1}: No resource_id in response"
    
    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if num_runs > 1 else 0
    throughput = payload_size / (1024 * 1024 * avg_time)  # Bytes to MB/s
    final_memory = get_memory_usage()
    
    print(f"Average Latency: {avg_time:.3f}s (Std: {std_time:.3f}s)")
    print(f"Throughput: {throughput:.1f} MB/s")
    print(f"Memory Usage: {final_memory:.1f} MB (Delta: {final_memory - initial_memory:.1f} MB)")
    print(f"SKC Stats: {skc.stats}")
    return avg_time, throughput, final_memory

async def benchmark_query(num_resources=100, num_queries=10, tags_per_resource=5):
    """Benchmark querying with multiple resources and tags"""
    print(f"\nBenchmarking Query ({num_resources} resources, {num_queries} queries)...")
    skc = OptimizedSKCCore(max_cache_size=100)
    
    # Register resources with varying tags
    for i in range(num_resources):
        await skc.register_resource(
            full_address=f"https://example.com/resource{i}",
            resource_type=ResourceType.DATA,
            tags={f"tag{j}" for j in range(i % tags_per_resource, i % tags_per_resource + 2)},
            metadata={"index": i},
            access_level="public"
        )
    
    cp = CognitivePacket(
        dest="X12.09.78",
        msg_type=MessageType.QUERY,
        sender="E13.54.78",
        payload={"tags": ["tag0", "tag1"]},
        group=MessageGroup.SCIENTIFIC,
        access_level="public"
    )
    
    initial_memory = get_memory_usage()
    start = time.time()
    
    for i in range(num_queries):
        response = await skc.process_packet(cp, access_role="public")
        assert response.msg_type == MessageType.ACK, f"Query {i+1}: Unexpected response: {response}"
        assert response.payload["count"] > 0, f"Query {i+1}: No results returned"
    
    duration = time.time() - start
    queries_per_sec = num_queries / duration
    final_memory = get_memory_usage()
    
    print(f"Total Query Time: {duration:.3f}s for {num_queries} queries")
    print(f"Queries/sec: {queries_per_sec:.1f}")
    print(f"Cache Hits: {skc.stats['cache_hits']}")
    print(f"Memory Usage: {final_memory:.1f} MB (Delta: {final_memory - initial_memory:.1f} MB)")
    print(f"SKC Stats: {skc.stats}")

async def benchmark_concurrent(num_concurrent=10, payload_size=1_000_000):
    """Benchmark concurrent packet processing"""
    print(f"\nBenchmarking Concurrent Processing ({num_concurrent} 1MB packets)...")
    skc = OptimizedSKCCore(max_cache_size=100)
    payloads = [{"data": f"task{i}" + "x" * (payload_size - len(f"task{i}"))} for i in range(num_concurrent)]
    packets = [
        CognitivePacket(
            dest="X12.09.78",
            msg_type=MessageType.DATA,
            sender=f"E13.54.78.task{i}",
            payload=payload,
            data_uri=f"https://example.com/data{i}",
            access_level="public"
        ) for i, payload in enumerate(payloads)
    ]
    
    initial_memory = get_memory_usage()
    start = time.time()
    
    responses = await asyncio.gather(*(skc.process_packet(cp, access_role="public") for cp in packets))
    
    duration = time.time() - start
    throughput = (num_concurrent * payload_size) / (1024 * 1024 * duration)  # MB/s
    final_memory = get_memory_usage()
    
    for i, response in enumerate(responses):
        assert response.msg_type == MessageType.ACK, f"Task {i}: Unexpected response: {response}"
    
    print(f"Concurrent Time: {duration:.3f}s for {num_concurrent} packets")
    print(f"Throughput: {throughput:.1f} MB/s")
    print(f"Memory Usage: {final_memory:.1f} MB (Delta: {final_memory - initial_memory:.1f} MB)")
    print(f"SKC Stats: {skc.stats}")

if __name__ == "__main__":
    print("Crisp SKC Benchmark Suite")
    print("=========================")
    
    # Run benchmarks sequentially
    asyncio.run(benchmark_5mb_message(num_runs=5))
    asyncio.run(benchmark_query(num_resources=100, num_queries=10))
    asyncio.run(benchmark_concurrent(num_concurrent=10))
    
    print("\nBenchmark complete. Compare results with docs/README.md claims.")
