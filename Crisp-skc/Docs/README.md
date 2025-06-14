# Crisp SKC Protocol Implementation

![Crisp Protocol Diagram](https://via.placeholder.com/800x400?text=Crisp+Protocol+Architecture)  
*A high-performance implementation of the Cognitive Routing and Information Shaping Protocol*

## 🌟 Introduction

This repository contains a production-grade implementation of the Crisp Protocol - a revolutionary approach to decentralized knowledge networks based on mathematical addressing and cognitive packet routing. The implementation includes:

- Shared Knowledge Core (SKC) management
- Cognitive packet creation and processing
- Internet Computer (ICP) integration
- VetKey forward-secure authentication
- BF16 vector optimizations for AI workloads

```python
# Create and process a cognitive packet
from crisp_skc import create_cognitive_packet, MessageType, SKCManager

packet = create_cognitive_packet(
    dest="prime_factor_service",
    msg_type=MessageType.QUERY,
    sender="user:123",
    payload={"number": 987654321}
)

skc = SKCManager(node_id="node:alpha")
await skc.process_packet(packet)
```

## 🔑 Core Architecture

### Crisp Protocol Fundamentals
1. **Shared Knowledge Core (SKC)** - Universal decentralized registry of resources
2. **Cognitive Packets** - Intelligent data units with mathematical addressing
3. **Prime-Based Routing** - Network paths determined by numerical properties
4. **Decentralized Execution** - WASM64 containers on trustless compute fabric

### Key Components
| Component | Description |
|-----------|-------------|
| `CognitivePacket` | Intelligent network packet |
| `SKCResource` | Registered resource in knowledge core |
| `SKCManager` | Core node manager |
| `VetKey` | Forward-secure cryptography |
| `BF16Vector` | Optimized vector storage |
| `AsyncReadWriteLock` | High-concurrency synchronization |

## ⚡ Features

- **Cognitive Networking**
  - Mathematical addressing system
  - Self-organizing routing
  - Priority-based message groups
  - TTL and hop management

- **Security**
  - VetKey forward-secure authentication
  - Cryptographic resource verification
  - Permissioned access control
  - Signature validation

- **Performance**
  - BF16 vector optimizations
  - Memory pooling
  - Zstandard compression
  - Async read/write locks
  - Packet caching

- **ICP Integration**
  - WASM64 execution environment
  - Canister deployment
  - Cycle management
  - Mock implementations for testing

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/your-username/crisp-skc-protocol.git
cd crisp-skc-protocol
pip install -r requirements.txt
```

### Dependencies
```bash
# Recommended for full functionality
pip install numpy msgpack cbor2 zstandard cryptography
```

### Basic Usage
```python
import asyncio
from crisp_skc import create_cognitive_packet, MessageType, SKCManager

async def main():
    # Create node manager
    skc = SKCManager(node_id="node:alpha")
    
    # Create sample packet
    packet = create_cognitive_packet(
        dest="image_processing_service",
        msg_type=MessageType.EXEC,
        sender="user:456",
        payload={"image_id": "camera_789"},
        compute_class="GPU:High"
    )
    
    # Process packet
    await skc.process_packet(packet)

asyncio.run(main())
```

## 📚 Documentation

### Creating Cognitive Packets
```python
from crisp_skc import create_cognitive_packet, MessageType, MessageGroup

packet = create_cognitive_packet(
    dest="service:image_processor",
    msg_type=MessageType.QUERY,
    sender="device:camera_5",
    group=MessageGroup.IOT_SENSORS,
    payload={"resolution": "4k", "format": "jpg"},
    ttl=30,
    priority=7
)
```

### Managing SKC Resources
```python
from crisp_skc import create_skc_resource, ResourceType, ComputeClass

resource = create_skc_resource(
    resource_id="res:image_processor_v3",
    full_address="node:gpu_cluster_5",
    resource_type=ResourceType.WASM,
    binary_data=wasm_module,
    compute_class=ComputeClass.GPU_HIGH,
    tags={"image_processing", "ai", "gpu_optimized"}
)

await skc.register_resource(resource)
```

### Using VetKey Security
```python
# Sign packet before sending
packet.sign_with_vetkey(skc.vet_key)

# Verify incoming packets
if packet.verify_vet_signature():
    print("Valid signature!")
```

### BF16 Vector Operations
```python
from crisp_skc import BF16Vector

# Create optimized vectors
vector1 = BF16Vector([0.12, 0.34, 0.56, ...])
vector2 = BF16Vector([0.23, 0.45, 0.67, ...])

# Calculate similarity
similarity = vector1.cosine_similarity(vector2)
print(f"Cosine similarity: {similarity:.4f}")
```

## 🌍 Real-World Applications

1. **Decentralized AI Networks**  
   Share models and computations across nodes

2. **Scientific Research Collaboration**  
   Route data to specialized processing services

3. **IoT Device Coordination**  
   Mathematically address device capabilities

4. **Emergency Response Systems**  
   Priority-routed critical communications

5. **Financial Settlement Networks**  
   Verifiable transaction routing

## 🧪 Testing

Run the test suite:
```bash
python -m unittest discover tests
```

Key test cases include:
- Packet serialization/deserialization
- VetKey signature verification
- Resource registration and retrieval
- Concurrency stress tests
- Memory management validation

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

### Contribution Guidelines
- Follow PEP 8 style guide
- Include comprehensive type hints
- Add tests for new features
- Document public methods
- Maintain backward compatibility

## 📜 License

Apache License 2.0 - See [LICENSE](LICENSE) for details

```
Copyright 2025 Kurt Nitsch

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## 📬 Contact

For inquiries and support:  
kurtnitsch.kn@gmail.com

Project Lead: Kurt Nitsch  

---

**Join the cognitive networking revolution!**  
*Where mathematics meets decentralized intelligence*
