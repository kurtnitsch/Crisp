#!/usr/bin/env python3
"""
Enhanced Crisp SKC Library with SmallPond Environment for AI Agents
- Memory optimizations for large-scale operations
- Granular error handling
- Thread safety enhancements
- Vector operations optimization
- Resource cleanup mechanisms
- Additional validation layers
- SmallPond environment for AI agent interaction
"""

import asyncio
import hashlib
import time
import uuid
import base64
import struct
import logging
import json
import os
import math
import dataclasses
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from functools import lru_cache, cached_property

# Configure logging with better performance
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom exceptions for better error handling
class SerializationError(Exception):
    """Custom exception for serialization failures"""
    pass

class ValidationError(Exception):
    """Custom exception for data validation failures"""
    pass

class CryptoError(Exception):
    """Custom exception for cryptographic operations"""
    pass

class PondError(Exception):
    """Custom exception for SmallPond operations"""
    pass

# Try to import NumPy for vector optimizations
NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    logger.info("NumPy available for vector optimizations.")
except ImportError:
    logger.warning("NumPy not available. Vector operations will use fallback implementations.")

# Try to import msgpack with fallback
MSGPACK_AVAILABLE = False
msgpack = None
try:
    import msgpack
    MSGPACK_AVAILABLE = True
    logger.info("msgpack available.")
except ImportError:
    logger.warning("msgpack not available, using json as fallback for serialization. Performance may be impacted.")
    
    # Create msgpack-like interface using json
    class MsgpackFallback:
        class exceptions:
            class PackException(Exception): pass
            class UnpackException(Exception): pass
        
        @staticmethod
        def packb(data, use_bin_type=True):
            try:
                def serialize_helper(obj):
                    if isinstance(obj, bytes):
                        return {"__bytes__": base64.b64encode(obj).decode('ascii')}
                    elif isinstance(obj, dict):
                        return {k: serialize_helper(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [serialize_helper(item) for item in obj]
                    return obj
                
                serializable_data = serialize_helper(data)
                return json.dumps(serializable_data, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            except (TypeError, ValueError) as e:
                logger.error(f"JSON packb failed: {e}. Data type: {type(data)}")
                raise MsgpackFallback.exceptions.PackException(f"JSON packb failed: {e}") from e
        
        @staticmethod
        def unpackb(data, raw=False, strict_map_key=False):
            try:
                def deserialize_helper(obj):
                    if isinstance(obj, dict) and "__bytes__" in obj:
                        try:
                            return base64.b64decode(obj["__bytes__"])
                        except Exception as e:
                            logger.error(f"Base64 decoding failed: {e}")
                    elif isinstance(obj, dict):
                        return {k: deserialize_helper(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [deserialize_helper(item) for item in obj]
                    return obj
                
                json_data = json.loads(data.decode('utf-8'))
                return deserialize_helper(json_data)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
                logger.error(f"JSON unpackb failed: {e}. Data (first 100 bytes): {data[:100]}")
                raise MsgpackFallback.exceptions.UnpackException(f"JSON unpackb failed: {e}") from e
    
    msgpack = MsgpackFallback()

# Import ICP agent components with fallbacks
ICP_AVAILABLE = False
Principal = HttpAgent = Identity = AgentError = None
try:
    from ic_agent import Principal, HttpAgent, Identity
    from ic_agent.errors import AgentError
    ICP_AVAILABLE = True
    logger.info("ICP agent libraries loaded successfully.")
except ImportError:
    logger.warning("ICP agent libraries not available, using mock implementations. ICP features will be simulated.")
    
    # Mock implementations for ICP components
    class Principal:
        def __init__(self, text: str):
            if not text or not isinstance(text, str):
                raise ValueError("Principal text cannot be empty or non-string.")
            self.text = text
        
        @classmethod
        def from_text(cls, text: str):
            if not text or not isinstance(text, str):
                raise ValueError("Invalid principal text format.")
            if not all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in text.lower()):
                raise ValueError(f"Invalid principal format: {text}")
            return cls(text)
        
        def __str__(self):
            return self.text
            
        def __repr__(self):
            return f"Principal('{self.text}')"
            
        def __eq__(self, other):
            if isinstance(other, Principal):
                return self.text == other.text
            return False
            
        def __hash__(self):
            return hash(self.text)
    
    class HttpAgent:
        def __init__(self, endpoint: str):
            if not endpoint:
                raise ValueError("HttpAgent endpoint cannot be empty")
            self.endpoint = endpoint
            logger.info(f"Mock ICP HttpAgent initialized with endpoint: {endpoint}")
        
        async def update_canister(self, canister_id: Principal, binary_data: bytes, identity=None):
            logger.warning(f"Mock ICP deployment: Simulating deployment to {canister_id}.")
            if not isinstance(canister_id, Principal):
                raise TypeError("canister_id must be a Principal object.")
            if not isinstance(binary_data, bytes):
                raise TypeError("binary_data must be bytes.")
            if len(binary_data) == 0:
                raise ValueError("binary_data cannot be empty for deployment.")
            await asyncio.sleep(0.05)
            logger.info(f"Mock deployment to {canister_id} completed.")
            return {"status": "success", "canister_id": str(canister_id)}
            
        async def install_code(self, canister_id: Principal, wasm_module: bytes, args: bytes = b'', mode: str = "install", identity=None):
            logger.warning(f"Mock ICP install_code: Simulating install to {canister_id}.")
            if not isinstance(canister_id, Principal) or not isinstance(wasm_module, bytes):
                raise TypeError("Invalid types for mock install_code.")
            if len(wasm_module) == 0:
                raise ValueError("wasm_module cannot be empty for installation.")
            await asyncio.sleep(0.05)
            logger.info(f"Mock install_code to {canister_id} completed.")
            return {"status": "installed", "canister_id": str(canister_id)}

    class Identity:
        def __init__(self, principal_text: str = "aaaaa-aa"):
            try:
                self._principal = Principal.from_text(principal_text)
            except ValueError as e:
                logger.error(f"Failed to create Identity with principal '{principal_text}': {e}")
                self._principal = Principal.from_text("aaaaa-aa")
        
        def get_principal(self):
            return self._principal
        
        def sign(self, blob: bytes) -> bytes:
            if not isinstance(blob, bytes):
                raise TypeError("blob must be bytes")
            logger.debug("Mock signing: Returning a dummy signature.")
            return b"mock_signature_" + hashlib.sha256(blob).digest()[:8]
            
        def __repr__(self):
            return f"Identity('{self.get_principal()}')"
    
    class AgentError(Exception):
        """Mock AgentError for ICP-related exceptions."""
        pass

# Try to import cryptography with fallback
CRYPTO_AVAILABLE = False
BasicIdentity = InvalidSignature = None
try:
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.backends import default_backend
    from cryptography.exceptions import InvalidSignature
    CRYPTO_AVAILABLE = True
    logger.info("Cryptography library available.")
    
    class BasicIdentity(Identity):
        def __init__(self, private_key=None, principal_text: str = None):
            self._private_key = None
            if private_key and isinstance(private_key, ed25519.Ed25519PrivateKey):
                self._private_key = private_key
                super().__init__(principal_text or "crypto-derived-principal")
            else:
                super().__init__(principal_text or "mock-principal-unsupported-key")
        
        @staticmethod
        def from_pem(pem_bytes: bytes):
            if not isinstance(pem_bytes, bytes):
                raise TypeError("pem_bytes must be bytes")
            try:
                private_key = serialization.load_pem_private_key(pem_bytes, password=None)
                if not isinstance(private_key, ed25519.Ed25519PrivateKey):
                    logger.warning("Unsupported private key type")
                    return BasicIdentity(principal_text="mock-principal-unsupported-key")
                return BasicIdentity(private_key=private_key)
            except Exception as e:
                logger.error(f"Failed to load identity from PEM: {e}")
                return BasicIdentity(principal_text="mock-principal-pem-fail")
        
        def sign(self, blob: bytes) -> bytes:
            if not isinstance(blob, bytes):
                raise TypeError("blob must be bytes")
            if self._private_key:
                try:
                    return self._private_key.sign(blob)
                except Exception as e:
                    logger.error(f"Signing failed: {e}")
                    raise CryptoError(f"Signing failed: {e}") from e
            return super().sign(blob)
    
except ImportError:
    logger.warning("Cryptography library not available, using basic mock identity.")
    
    class BasicIdentity(Identity):
        def __init__(self, principal_text: str = "mock-principal"):
            super().__init__(principal_text)
        
        @staticmethod
        def from_pem(pem_bytes: bytes):
            if not isinstance(pem_bytes, bytes):
                raise TypeError("pem_bytes must be bytes")
            logger.warning("Mock BasicIdentity.from_pem: Cryptography library not available.")
            return BasicIdentity("mock-principal-from-pem")
    
    class InvalidSignature(Exception): pass

# Try to import requests for error handling
REQUESTS_AVAILABLE = False
try:
    import requests.exceptions
    REQUESTS_AVAILABLE = True
except ImportError:
    logger.warning("requests library not available. Some network-related error handling might be less granular.")
    class requests:
        class exceptions:
            class RequestException(Exception): pass

# Try to import cbor2
CBOR2_AVAILABLE = False
try:
    import cbor2
    CBOR2_AVAILABLE = True
    logger.info("cbor2 available.")
except ImportError:
    logger.warning("cbor2 not available. CBOR serialization will be disabled.")
    class cbor2:
        @staticmethod
        def dumps(data): raise NotImplementedError("cbor2 library not installed")
        @staticmethod
        def loads(data): raise NotImplementedError("cbor2 library not installed")

# Try to import zstandard
ZSTD_AVAILABLE = False
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
    logger.info("zstandard available.")
except ImportError:
    logger.warning("zstandard not available. Zstd compression will be disabled.")
    class zstd:
        class ZstdCompressor:
            def compress(self, data): return data
        class ZstdDecompressor:
            def decompress(self, data): return data

# ============================================================================
# Constants
# ============================================================================
ICP_WASM64_MAX_MEM = 4 * 1024 * 1024 * 1024  # 4GiB
MAX_HOPS = 15
MAX_PACKET_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1MB
MAX_POND_AGENTS = 50  # Limit for SmallPond environment
POND_TICK_INTERVAL = 0.1  # 100ms

# ============================================================================
# New Feature Classes (VetKey, BF16Vector, MemoryPool)
# ============================================================================

class VetKey:
    """VetKey implementation for forward-secure key derivation and signing."""
    def __init__(self):
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography library is required for VetKey.")
        self._signing_key = ed25519.Ed25519PrivateKey.generate()
        self._exchange_key = x25519.X25519PrivateKey.generate()
        self.epoch = 0
        self.chain_key = os.urandom(32)

    @property
    def signing_public_key(self) -> ed25519.Ed25519PublicKey:
        return self._signing_key.public_key()

    def signing_public_bytes(self) -> bytes:
        return self.signing_public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

    def derive_shared_secret(self, peer_exchange_public_key_bytes: bytes) -> bytes:
        peer_key = x25519.X25519PublicKey.from_public_bytes(peer_exchange_public_key_bytes)
        shared_secret = self._exchange_key.exchange(peer_key)
        
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'Crisp-VetKey-Derivation',
            backend=default_backend()
        ).derive(shared_secret)

    def ratchet_forward(self) -> None:
        """Advance the key state for forward secrecy."""
        self.epoch += 1
        self.chain_key = hashlib.blake2b(b'ratchet-salt' + self.chain_key, digest_size=32).digest()

    def sign(self, data: bytes) -> bytes:
        """Sign data using the Ed25519 signing key."""
        return self._signing_key.sign(data)

    @staticmethod
    def verify(signature: bytes, data: bytes, public_key_bytes: bytes) -> bool:
        """Verify a signature using a public key."""
        try:
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
            public_key.verify(signature, data)
            return True
        except InvalidSignature:
            logger.debug("VetKey signature verification failed: InvalidSignature.")
            return False
        except Exception as e:
            logger.debug(f"VetKey signature verification failed: {e}")
            return False

class MockVetKey:
    """Mock VetKey for when cryptography is not available."""
    def __init__(self):
        self.epoch = 0
    def signing_public_bytes(self) -> bytes: return b'\x00' * 32
    def ratchet_forward(self) -> None: self.epoch += 1
    def sign(self, data: bytes) -> bytes: return b"mock_vetkey_sig_" + hashlib.sha256(data).digest()[:8]
    @staticmethod
    def verify(signature: bytes, data: bytes, public_key_bytes: bytes) -> bool: return True

class BF16Vector:
    """Efficient storage for vectors using Brain Float 16 format."""
    def __init__(self, data: Optional[Union[List[float], np.ndarray]] = None):
        if NUMPY_AVAILABLE and isinstance(data, np.ndarray) and data.dtype == np.float32:
            self._data = self._convert_numpy_to_bf16(data)
        elif data:
            self._data = self._convert_to_bf16(data)
        else:
            self._data = bytearray()

    @staticmethod
    def _convert_numpy_to_bf16(array: np.ndarray) -> bytearray:
        """Optimized conversion using NumPy vectorization"""
        int_reps = array.view(np.uint32)
        bf16_reps = (int_reps >> 16).astype(np.uint16)
        return bytearray(bf16_reps.tobytes())

    @staticmethod
    def _convert_to_bf16(floats: List[float]) -> bytearray:
        """Convert float32 list to BF16 bytes."""
        bf16_data = bytearray()
        for f in floats:
            try:
                int_rep = struct.unpack('!I', struct.pack('!f', f))[0]
                bf16_rep = (int_rep >> 16) & 0xFFFF
                bf16_data.extend(struct.pack('!H', bf16_rep))
            except (struct.error, TypeError):
                logger.warning(f"Could not convert value {f} to float, skipping.")
        return bf16_data

    def to_float32(self) -> List[float]:
        """Convert back to float32 list."""
        floats = []
        for i in range(0, len(self._data), 2):
            bf16_val = struct.unpack('!H', self._data[i:i+2])[0]
            float_val = struct.unpack('!f', struct.pack('!I', bf16_val << 16))[0]
            floats.append(float_val)
        return floats

    def cosine_similarity(self, other: 'BF16Vector') -> float:
        """Compute cosine similarity using NumPy if available"""
        if len(self) != len(other):
            raise ValueError("Vectors must have same dimensions for cosine similarity")
        
        if NUMPY_AVAILABLE:
            a = np.array(self.to_float32(), dtype=np.float32)
            b = np.array(other.to_float32(), dtype=np.float32)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        
        # Fallback to pure Python
        a = self.to_float32()
        b = other.to_float32()
        dot = sum(x*y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x*x for x in a))
        norm_b = math.sqrt(sum(y*y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @property
    def bf16_data(self) -> bytes:
        return bytes(self._data)

    @bf16_data.setter
    def bf16_data(self, value: bytes) -> None:
        if len(value) % 2 != 0:
            raise ValueError("BF16 data must be an even number of bytes.")
        self._data = bytearray(value)

    def __len__(self) -> int:
        return len(self._data) // 2

    def size_bytes(self) -> int:
        return len(self._data)

class MemoryPool:
    """Reusable memory pool for efficient buffer management"""
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self._pool = []
        self._chunk_size = chunk_size
    
    def get_buffer(self, size: int) -> bytearray:
        """Get a buffer of at least the requested size"""
        for buf in self._pool:
            if len(buf) >= size:
                self._pool.remove(buf)
                return buf
        return bytearray(max(size, self._chunk_size))
    
    def return_buffer(self, buf: bytearray) -> None:
        """Return buffer to pool for reuse"""
        buf[:] = b''
        self._pool.append(buf)

# ============================================================================
# Core Data Structures and Enums
# ============================================================================

class MessageType(Enum):
    """Cognitive Packet message types"""
    QUERY = "query"
    DATA = "data"
    EXEC = "exec"
    COMMAND = "command"
    ACK = "ack"
    ERROR = "error"
    UPDATE = "update"
    DISCOVER = "discover"
    COMPOSE = "compose"
    ADAPT = "adapt"
    VALIDATE = "validate"
    OPTIMIZE = "optimize"
    POND = "pond"  # Special type for SmallPond communication

class ComputeClass(Enum):
    """Compute resource classifications"""
    CPU_LOW = "CPU:Low"
    CPU_MEDIUM = "CPU:Medium"
    CPU_HIGH = "CPU:High"
    GPU_LOW = "GPU:Low"
    GPU_MEDIUM = "GPU:Medium"
    GPU_HIGH = "GPU:High"
    WASM_SANDBOXED = "WASM:Sandboxed"
    WASM64 = "WASM:64Bit"
    WASM64_ICP = "WASM64:ICP"
    TPU_AVAILABLE = "TPU:Available"
    FPGA = "FPGA"

class ResourceType(Enum):
    """SKC resource types"""
    DATA = "data"
    MODEL = "model"
    WASM = "wasm"
    EXECUTABLE = "executable"
    TELEMETRY = "telemetry"
    ONTOLOGY = "ontology"
    EMBEDDING = "embedding"
    ALGORITHM = "algorithm"
    WORKFLOW = "workflow"
    SCHEMA = "schema"

class MessageGroup(Enum):
    """Message group classifications"""
    STANDARD = "Standard"
    HIGH_PRIORITY = "High Priority"
    FINANCE = "Finance"
    MEDICAL = "Medical"
    IOT_SENSORS = "IoT/Sensors"
    MILITARY = "Military"
    ROBOTICS = "Robotics"
    SCIENTIFIC = "Scientific Research"
    SECURITY = "Security"
    EMERGENCY = "Emergency"
    AI_ML = "AI/Machine Learning"
    POND = "Pond"  # Special group for SmallPond messages

# ============================================================================
# Fixed Asynchronous Read-Write Lock
# ============================================================================

class AsyncReadWriteLock:
    """
    An asynchronous Read-Write Lock for asyncio.
    Allows multiple readers or one writer. Writers are prioritized.
    """
    def __init__(self):
        self._lock = asyncio.Lock()
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False
        self._no_writers = asyncio.Event()
        self._reader_can_proceed = asyncio.Event()
        self._no_writers.set()
        self._reader_can_proceed.set()

    async def reader_acquire(self):
        async with self._lock:
            while self._writers_waiting > 0 or self._writer_active:
                self._reader_can_proceed.clear()
                await self._reader_can_proceed.wait()
            self._readers += 1
            if self._readers == 1:
                self._no_writers.clear()

    async def reader_release(self):
        async with self._lock:
            if self._readers <= 0:
                logger.warning("reader_release called with no active readers")
                return
            self._readers -= 1
            if self._readers == 0:
                self._no_writers.set()

    async def writer_acquire(self):
        async with self._lock:
            self._writers_waiting += 1
            if self._writers_waiting > 0:
                self._reader_can_proceed.clear()
        
        try:
            await self._no_writers.wait()
            async with self._lock:
                self._writers_waiting -= 1
                self._writer_active = True
                self._no_writers.clear()
        except asyncio.CancelledError:
            async with self._lock:
                self._writers_waiting -= 1
                if self._writers_waiting == 0 and not self._writer_active:
                    self._reader_can_proceed.set()
            raise

    async def writer_release(self):
        async with self._lock:
            if not self._writer_active:
                logger.warning("writer_release called with no active writer")
                return
            self._writer_active = False
            if self._writers_waiting == 0:
                self._reader_can_proceed.set()
            self._no_writers.set()

    class ReaderContext:
        def __init__(self, rw_lock):
            self._rw_lock = rw_lock
        
        async def __aenter__(self):
            await self._rw_lock.reader_acquire()
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self._rw_lock.reader_release()

    class WriterContext:
        def __init__(self, rw_lock):
            self._rw_lock = rw_lock
        
        async def __aenter__(self):
            await self._rw_lock.writer_acquire()
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self._rw_lock.writer_release()

    @property
    def reader(self):
        return self.ReaderContext(self)

    @property
    def writer(self):
        return self.WriterContext(self)

# ============================================================================
# Optimized Data Classes
# ============================================================================

@dataclass
class CognitivePacket:
    """
    Represents a self-aware, intelligent unit of information and intent.
    """
    # Core required fields
    dest: str 
    msg_type: MessageType
    sender: str

    # Auto-generated fields
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Optional fields with proper defaults
    alias: Optional[str] = None
    ttl: Optional[int] = None
    hops: int = 0
    priority: int = 5
    group: Optional[MessageGroup] = None
    part: Optional[str] = None
    stream_id: Optional[str] = None
    chain_id: Optional[str] = None

    # Content metadata
    size: Optional[str] = None
    encoding: str = "msgpack"
    compression: str = "none"
    data_type: Optional[str] = None
    _data_hash: Optional[str] = field(default=None, init=False, repr=False)
    data_uri: Optional[str] = None

    # Execution context
    compute_class: Optional[ComputeClass] = None
    context: Optional[str] = None
    exec_id: Optional[str] = None
    exec_status: Optional[str] = None
    exec_params: Optional[Dict[str, Any]] = None

    # Security
    auth_token: Optional[str] = None
    encryption: Optional[str] = None
    access_level: Optional[str] = None
    vet_signature: Optional[bytes] = None
    vet_public_key: Optional[bytes] = None
    vet_epoch: Optional[int] = None

    # Control flow
    next_action: Optional[str] = None
    retry_count: int = 0
    timeout: Optional[int] = None
    error_code: Optional[str] = None

    # AI-specific fields
    ai_model: Optional[str] = None
    ai_intent: Optional[str] = None
    ai_confidence: Optional[float] = None
    ai_feedback: Optional[str] = None
    ai_exec_env: Optional[str] = None

    # Audit trail
    log_id: Optional[str] = None
    modified_by: Optional[str] = None
    version: str = "1.0"
    change_timestamp: Optional[str] = None

    # ICP-specific fields
    canister_id: Optional[str] = None
    cycles_required: Optional[int] = None

    # Payload data
    payload: Optional[Dict[str, Any]] = None
    binary_payload: Optional[bytes] = None

    # Cached serialization
    _cached_binary: Optional[bytes] = field(default=None, init=False, repr=False)
    _cached_binary_valid: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        # Default encoding based on availability
        if self.encoding == 'msgpack' and not MSGPACK_AVAILABLE:
            self.encoding = 'cbor' if CBOR2_AVAILABLE else 'json'
        
        # Validate required fields
        if not self.dest: 
            raise ValidationError("dest field cannot be empty")
        if not self.sender: 
            raise ValidationError("sender field cannot be empty")
            
        # Ensure enums are proper
        if isinstance(self.msg_type, str):
            try: 
                self.msg_type = MessageType(self.msg_type.lower())
            except ValueError: 
                self.msg_type = MessageType.QUERY
        
        if isinstance(self.group, str):
            try: 
                self.group = MessageGroup(self.group)
            except ValueError: 
                self.group = None

        if isinstance(self.compute_class, str):
            try: 
                self.compute_class = ComputeClass(self.compute_class)
            except ValueError: 
                self.compute_class = None

        if self._data_hash is None and (self.payload is not None or self.binary_payload is not None):
            self._data_hash = self._compute_hash_fast()

    @property
    def data_hash(self) -> Optional[str]:
        if self._data_hash is None and (self.payload is not None or self.binary_payload is not None):
            self._data_hash = self._compute_hash_fast()
        return self._data_hash

    def _compute_hash_fast(self) -> str:
        try:
            hasher = hashlib.blake2b(digest_size=8)
            if self.binary_payload is not None:
                hasher.update(self.binary_payload)
            if self.payload is not None:
                sorted_json = json.dumps(self.payload, sort_keys=True, separators=(',', ':'))
                hasher.update(sorted_json.encode('utf-8'))
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Hash computation failed for packet {self.msg_id}: {e}")
            return ""

    def invalidate_cache(self) -> None:
        self._cached_binary_valid = False
        self._cached_binary = None
        self._data_hash = None
        self.vet_signature = None

    def to_binary_format(self) -> bytes:
        if self._cached_binary_valid and self._cached_binary is not None:
            return self._cached_binary

        try:
            headers = self._build_headers()
            
            # Select serializer
            if self.encoding.lower() == 'cbor' and CBOR2_AVAILABLE:
                header_data = cbor2.dumps(headers)
                payload_data = cbor2.dumps(self.payload) if self.payload is not None else b''
            elif MSGPACK_AVAILABLE:
                header_data = msgpack.packb(headers, use_bin_type=True)
                payload_data = msgpack.packb(self.payload, use_bin_type=True) if self.payload is not None else b''
            else:
                header_data = msgpack.packb(headers)
                payload_data = msgpack.packb(self.payload) if self.payload is not None else b''
                
            binary_data = self.binary_payload or b''
            
            # Validate sizes
            sections = [
                ("header", header_data, MAX_PACKET_SIZE),
                ("payload", payload_data, MAX_PACKET_SIZE),
                ("binary_payload", binary_data, MAX_PACKET_SIZE)
            ]
            
            for name, data, max_size in sections:
                data_len = len(data)
                if data_len > max_size:
                    raise SerializationError(f"{name} section exceeds maximum allowed size ({data_len} > {max_size})")
                if data_len > 2**32 - 1:
                    raise SerializationError(f"{name} section too large for binary format (>4GB)")
                
            # Pack into binary format
            result = struct.pack('!III', len(header_data), len(payload_data), len(binary_data))
            result += header_data + payload_data + binary_data

            self._cached_binary = result
            self._cached_binary_valid = True
            return result
        except struct.error as e:
            logger.error(f"Struct packing error: {e}")
            raise SerializationError("Data too large for binary format") from e
        except (OverflowError, TypeError) as e:
            logger.error(f"Data serialization error: {e}")
            raise SerializationError("Invalid data types in payload") from e
        except Exception as e:
            logger.exception("Unexpected serialization failure")
            raise SerializationError("Unknown serialization error") from e

    def _build_headers(self) -> Dict[str, Any]:
        headers = {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_') and v is not None and k not in ['payload', 'binary_payload']
        }
        # Convert enums to values
        if 'msg_type' in headers: 
            headers['msg_type'] = self.msg_type.value
        if 'group' in headers: 
            headers['group'] = self.group.value if self.group else None
        if 'compute_class' in headers: 
            headers['compute_class'] = self.compute_class.value if self.compute_class else None
        
        # Base64 encode byte fields
        if self.vet_signature: 
            headers['vet_signature'] = base64.b64encode(self.vet_signature).decode('ascii')
        if self.vet_public_key: 
            headers['vet_public_key'] = base64.b64encode(self.vet_public_key).decode('ascii')
        return headers

    @classmethod
    def from_binary_format(cls, binary_data: bytes) -> 'CognitivePacket':
        if len(binary_data) < 12:
            raise SerializationError(f"Invalid binary packet: too short ({len(binary_data)} bytes)")

        try:
            header_len, payload_len, binary_len = struct.unpack('!III', binary_data[:12])
            total_len = 12 + header_len + payload_len + binary_len
            if len(binary_data) != total_len:
                raise SerializationError(
                    f"Binary packet length mismatch. Expected {total_len}, got {len(binary_data)}"
                )

            offset = 12
            header_bytes = binary_data[offset:offset+header_len]
            offset += header_len
            payload_bytes = binary_data[offset:offset+payload_len] if payload_len > 0 else b''
            offset += payload_len
            binary_payload = binary_data[offset:offset+binary_len] if binary_len > 0 else b''
            
            # Detect serializer
            if CBOR2_AVAILABLE and len(header_bytes) > 0 and (header_bytes[0] >> 5) == 5:
                header_data = cbor2.loads(header_bytes)
                payload_data = cbor2.loads(payload_bytes) if payload_bytes else None
            elif MSGPACK_AVAILABLE:
                header_data = msgpack.unpackb(header_bytes, raw=False)
                payload_data = msgpack.unpackb(payload_bytes, raw=False) if payload_bytes else None
            else:
                header_data = msgpack.unpackb(header_bytes)
                payload_data = msgpack.unpackb(payload_bytes) if payload_bytes else None

            # Rebuild packet
            instance_kwargs = cls._build_kwargs_from_headers(header_data, payload_data, binary_payload)
            instance = cls(**instance_kwargs)
            instance._cached_binary = binary_data
            instance._cached_binary_valid = True
            instance.decompress_payload()
            return instance
        except Exception as e:
            logger.error(f"Unexpected error during binary deserialization: {e}")
            raise SerializationError(f"Failed to parse binary packet: {e}") from e

    @classmethod
    def _build_kwargs_from_headers(cls, headers: Dict, payload_data: Any, binary_payload: bytes) -> Dict:
        kwargs = headers.copy()
        kwargs['payload'] = payload_data
        kwargs['binary_payload'] = binary_payload or None
        
        # Decode base64 fields
        if 'vet_signature' in kwargs and kwargs['vet_signature']:
            kwargs['vet_signature'] = base64.b64decode(kwargs['vet_signature'])
        if 'vet_public_key' in kwargs and kwargs['vet_public_key']:
            kwargs['vet_public_key'] = base64.b64decode(kwargs['vet_public_key'])

        # Filter out keys not in the dataclass definition
        valid_keys = {f.name for f in dataclasses.fields(cls) if f.init}
        return {k: v for k, v in kwargs.items() if k in valid_keys}

    def compress_payload(self, method: str = "zstd"):
        if not self.binary_payload or not ZSTD_AVAILABLE or method != "zstd": 
            return
        try:
            original_size = len(self.binary_payload)
            self.binary_payload = zstd.ZstdCompressor().compress(self.binary_payload)
            self.compression = "zstd"
            self.invalidate_cache()
            logger.debug(f"Compressed payload {original_size} -> {len(self.binary_payload)} bytes")
        except Exception as e: 
            logger.error(f"Zstd compression failed: {e}")

    def decompress_payload(self):
        if self.compression != "zstd" or not self.binary_payload or not ZSTD_AVAILABLE: 
            return
        try:
            original_size = len(self.binary_payload)
            self.binary_payload = zstd.ZstdDecompressor().decompress(self.binary_payload)
            self.compression = "none"
            self.invalidate_cache()
            logger.debug(f"Decompressed payload {original_size} -> {len(self.binary_payload)} bytes")
        except Exception as e: 
            logger.error(f"Zstd decompression failed: {e}")

    def sign_with_vetkey(self, vet_key: Union[VetKey, MockVetKey]):
        self.invalidate_cache()
        self.vet_public_key = vet_key.signing_public_bytes()
        self.vet_epoch = vet_key.epoch
        self.vet_signature = None 
        data_to_sign = self.to_binary_format()
        self.vet_signature = vet_key.sign(data_to_sign)
        self.invalidate_cache()

    def verify_vet_signature(self) -> bool:
        if not self.vet_signature or not self.vet_public_key: 
            return False
        if not CRYPTO_AVAILABLE and isinstance(self.vet_public_key, bytes):
            return True  # Accept all in mock mode
        
        signature = self.vet_signature
        temp_packet = dataclasses.replace(self, vet_signature=None, _cached_binary=None, _cached_binary_valid=False)
        data_that_was_signed = temp_packet.to_binary_format()
        return VetKey.verify(signature, data_that_was_signed, self.vet_public_key)

@dataclass
class SKCResource:
    """
    A resource registered within the Shared Knowledge Core (SKC).
    """
    resource_id: str
    full_address: str
    resource_type: ResourceType
    version: str = "1.0.0"
    compute_class: Optional[ComputeClass] = None
    size_bytes: int = 0
    created: Optional[str] = None
    permissions: Dict[str, List[str]] = field(default_factory=lambda: {"read": ["public"], "write": ["admin"]})
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    data: Optional[Any] = None
    binary_data: Optional[bytes] = None
    canister_id: Optional[str] = None
    cycles_required: Optional[int] = None
    compression: str = "none"
    vector_data: Optional[BF16Vector] = None
    _checksum: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not self.created: 
            self.created = datetime.now(timezone.utc).isoformat()
        if isinstance(self.tags, list): 
            self.tags = set(self.tags)
        
        if isinstance(self.resource_type, str): 
            self.resource_type = ResourceType(self.resource_type)
        if isinstance(self.compute_class, str): 
            self.compute_class = ComputeClass(self.compute_class)

        if self.resource_type == ResourceType.EMBEDDING and self.data:
            if NUMPY_AVAILABLE and isinstance(self.data, np.ndarray):
                self.vector_data = BF16Vector(self.data)
            elif isinstance(self.data, list):
                self.vector_data = BF16Vector(self.data)
            self.data = None
        
        if self.size_bytes == 0: 
            self._calculate_size()
        if self._checksum is None: 
            self._checksum = self._compute_checksum_fast()

    def _calculate_size(self):
        size = 0
        if self.binary_data: 
            size += len(self.binary_data)
        if self.vector_data: 
            size += self.vector_data.size_bytes()
        if self.data: 
            size += len(json.dumps(self.data).encode('utf-8'))
        self.size_bytes = size

    @cached_property
    def checksum(self) -> str:
        if self._checksum is None: 
            self._checksum = self._compute_checksum_fast()
        return self._checksum

    def _compute_checksum_fast(self) -> str:
        hasher = hashlib.blake2b(digest_size=16)
        if self.binary_data: 
            hasher.update(self.binary_data)
        if self.vector_data: 
            hasher.update(self.vector_data.bf16_data)
        if self.data: 
            hasher.update(json.dumps(self.data, sort_keys=True).encode())
        return hasher.hexdigest()

    def invalidate_cache(self) -> None:
        if 'checksum' in self.__dict__: 
            del self.__dict__['checksum']
        self._checksum = None
        self._calculate_size()

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        if self.vector_data: 
            d['vector_data'] = base64.b64encode(self.vector_data.bf16_data).decode('ascii')
        if self.binary_data: 
            d['binary_data'] = base64.b64encode(self.binary_data).decode('ascii')
        d.pop('_checksum', None)
        d['checksum'] = self.checksum
        d['resource_type'] = self.resource_type.value
        if self.compute_class: 
            d['compute_class'] = self.compute_class.value
        d['tags'] = list(self.tags)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SKCResource':
        data = data.copy()
        if 'binary_data' in data and data['binary_data']: 
            data['binary_data'] = base64.b64decode(data['binary_data'])
        if 'vector_data' in data and data['vector_data']:
            vec = BF16Vector()
            vec.bf16_data = base64.b64decode(data['vector_data'])
            data['vector_data'] = vec
        
        valid_keys = {f.name for f in dataclasses.fields(cls) if f.init}
        init_kwargs = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**init_kwargs)

# ============================================================================
# Core SKC Manager and Network Components
# ============================================================================

class NetworkStats:
    def __init__(self): 
        self._lock = asyncio.Lock()
        self._stats = defaultdict(int)
        self._stats['start_time'] = time.time()
        
    async def increment(self, name: str, val: int=1): 
        async with self._lock: 
            self._stats[name] += val
            
    async def get_stats(self) -> Dict[str, Any]: 
        async with self._lock: 
            stats = self._stats.copy()
            stats['uptime'] = time.time() - stats['start_time']
            return stats

class PacketCache:
    def __init__(self, max_size: int = 1000, ttl: int = 300): 
        self.max_size = max_size
        self.default_ttl = ttl
        self._cache = {}
        self._lru = []
        self._lock = asyncio.Lock()
        
    async def get(self, packet_id: str) -> Optional[CognitivePacket]:
        async with self._lock:
            entry = self._cache.get(packet_id)
            if not entry:
                return None
            packet, expiry = entry
            if time.time() < expiry: 
                self._lru.remove(packet_id)
                self._lru.append(packet_id)
                return packet
            del self._cache[packet_id]
            self._lru.remove(packet_id)
            return None
            
    async def put(self, packet: CognitivePacket, ttl: Optional[int] = None):
        async with self._lock:
            if packet.msg_id in self._cache: 
                self._lru.remove(packet.msg_id)
            elif len(self._cache) >= self.max_size and self._lru:
                oldest_id = self._lru.pop(0)
                del self._cache[oldest_id]
                
            expiry = time.time() + (ttl or self.default_ttl)
            self._cache[packet.msg_id] = (packet, expiry)
            self._lru.append(packet.msg_id)
            
    async def clear_expired(self):
        async with self._lock:
            now = time.time()
            expired = [k for k, (_, exp) in self._cache.items() if now >= exp]
            for k in expired: 
                del self._cache[k]
                if k in self._lru:
                    self._lru.remove(k)

class ICPManager:
    def __init__(self, endpoint: str): 
        self.agent = HttpAgent(endpoint) if ICP_AVAILABLE else None
        self._temp_files = []

    async def deploy_wasm(self, packet: CognitivePacket) -> bool:
        try:
            if not packet.binary_payload:
                logger.error("No WASM payload to deploy")
                return False
            return True
        except AgentError as e:
            logger.error(f"ICP deployment failed: {e}")
            return False
        finally:
            for f in self._temp_files:
                try:
                    os.unlink(f)
                except OSError:
                    pass
            self._temp_files = []

class MessageRouter:
    def __init__(self): 
        self._routes: Dict[str, List] = defaultdict(list)
        self._lock = AsyncReadWriteLock()
        
    async def add_route(self, dest: str, endpoint: str, weight: float = 1.0):
        async with self._lock.writer: 
            self._routes[dest].append((endpoint, weight))
            
    async def get_next_endpoint(self, dest: str) -> Optional[str]:
        async with self._lock.reader:
            routes = self._routes.get(dest)
            if not routes:
                return None
            return routes[0][0]  # Simple round-robin

class SKCManager:
    def __init__(self, node_id: str, icp_endpoint: str = "https://ic0.app"):
        self.node_id = node_id
        self.resources: Dict[str, SKCResource] = {}
        self.packet_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.stats = NetworkStats()
        self.cache = PacketCache()
        self.icp_manager = ICPManager(icp_endpoint)
        self.router = MessageRouter()
        self._resource_lock = AsyncReadWriteLock()
        self._handler_lock = asyncio.Lock()
        self.vet_key = VetKey() if CRYPTO_AVAILABLE else MockVetKey()
        self.known_public_keys: Dict[str, bytes] = {}
        self.memory_pool = MemoryPool()
        logger.info(f"SKCManager initialized for node {node_id}")

    async def register_resource(self, resource: SKCResource) -> bool:
        async with self._resource_lock.writer: 
            self.resources[resource.resource_id] = resource
            return True
            
    async def get_resource(self, resource_id: str) -> Optional[SKCResource]:
        async with self._resource_lock.reader: 
            return self.resources.get(resource_id)
            
    async def register_handler(self, msg_type: MessageType, handler: Callable):
        async with self._handler_lock: 
            self.packet_handlers[msg_type].append(handler)

    def validate_packet(func: Callable):
        async def wrapper(skc: 'SKCManager', packet: CognitivePacket):
            # Validate signature
            if packet.vet_signature and packet.vet_public_key:
                if not packet.verify_vet_signature():
                    logger.warning(f"VetKey signature verification failed for packet {packet.msg_id}")
                    await skc.stats.increment('errors_auth')
                    return False
                    
            # Validate TTL/hop count
            if packet.hops > MAX_HOPS:
                logger.warning(f"Packet TTL exceeded: {packet.msg_id} has {packet.hops} hops")
                await skc.stats.increment('errors_ttl_exceeded')
                return False
                
            # Validate size
            try:
                if len(packet.to_binary_format()) > MAX_PACKET_SIZE:
                    logger.warning(f"Packet too large: {packet.msg_id}")
                    await skc.stats.increment('errors_size_exceeded')
                    return False
            except SerializationError:
                return False
                
            return await func(skc, packet)
        return wrapper

    @validate_packet
    async def process_packet(self, packet: CognitivePacket) -> bool:
        await self.stats.increment('packets_received')
        
        # Check Cache
        cached = await self.cache.get(packet.msg_id)
        if cached:
            await self.stats.increment('cache_hits')
            return True
        await self.stats.increment('cache_misses')

        # Process with handlers
        handlers = self.packet_handlers.get(packet.msg_type, [])
        if not handlers:
            logger.debug(f"No handler for message type {packet.msg_type.value}, caching packet")
            await self.cache.put(packet)
            return True
            
        results = await asyncio.gather(
            *(h(packet) for h in handlers), 
            return_exceptions=True
        )
        
        success = True
        for r in results:
            if isinstance(r, Exception) or r is False:
                success = False
                logger.error(f"Handler error: {r}")
        
        if success:
            await self.cache.put(packet)
            logger.debug(f"Packet {packet.msg_id} processed successfully")
        else:
            logger.error(f"Packet {packet.msg_id} failed processing")
            await self.stats.increment('errors_processing')

        return success

    async def send_packet(self, packet: CognitivePacket) -> bool:
        endpoint = await self.router.get_next_endpoint(packet.dest)
        if not endpoint:
            logger.warning(f"No route for destination {packet.dest}")
            await self.stats.increment('errors_routing')
            return False
        
        # Sign packet
        packet.sign_with_vetkey(self.vet_key)
        self.vet_key.ratchet_forward()
        
        try:
            binary_data = packet.to_binary_format()
            buffer = self.memory_pool.get_buffer(len(binary_data))
            buffer[:] = binary_data
            
            await self.stats.increment('packets_sent')
            await self.stats.increment('bytes_sent', len(binary_data))

            logger.info(f"Sending packet {packet.msg_id} to {packet.dest} via {endpoint}")
            # Simulated network send
            await asyncio.sleep(0.01)
            return True
        except Exception as e:
            logger.error(f"Packet send failed: {e}")
            await self.stats.increment('errors_send_failed')
            return False
        finally:
            self.memory_pool.return_buffer(buffer)

# ============================================================================
# SmallPond Environment for AI Agents
# ============================================================================

class AgentAction:
    """Base class for agent actions in the SmallPond environment"""
    def __init__(self, name: str, cost: int = 1):
        self.name = name
        self.cost = cost
    
    async def execute(self, agent: 'PondAgent'):
        """Execute the action in the context of an agent"""
        raise NotImplementedError("Subclasses must implement execute method")
        
    def __str__(self):
        return f"Action({self.name}, cost={self.cost})"

class MoveAction(AgentAction):
    """Action for moving agents within the pond"""
    def __init__(self, direction: str):
        super().__init__(f"move_{direction}", cost=2)
        self.direction = direction
        
    async def execute(self, agent: 'PondAgent'):
        # Simple movement in a 2D grid
        new_x, new_y = agent.position
        if self.direction == "north": new_y -= 1
        elif self.direction == "south": new_y += 1
        elif self.direction == "east": new_x += 1
        elif self.direction == "west": new_x -= 1
            
        # Boundary check
        if 0 <= new_x < agent.pond.size and 0 <= new_y < agent.pond.size:
            agent.position = (new_x, new_y)
            return f"Moved {self.direction} to ({new_x}, {new_y})"
        return f"Could not move {self.direction} - boundary reached"

class CommunicateAction(AgentAction):
    """Action for sending messages between agents"""
    def __init__(self, recipient_id: str, message: dict):
        super().__init__(f"communicate_to_{recipient_id}", cost=3)
        self.recipient_id = recipient_id
        self.message = message
        
    async def execute(self, agent: 'PondAgent'):
        if self.recipient_id not in agent.pond.agents:
            return f"Recipient {self.recipient_id} not found"
            
        await agent.send_message(self.recipient_id, self.message)
        return f"Message sent to {self.recipient_id}"

class SenseAction(AgentAction):
    """Action for sensing the environment"""
    def __init__(self, radius: int = 1):
        super().__init__(f"sense_radius_{radius}", cost=1)
        self.radius = radius
        
    async def execute(self, agent: 'PondAgent'):
        # Detect nearby agents and resources
        x, y = agent.position
        nearby = []
        for other_id, other_agent in agent.pond.agents.items():
            if other_id == agent.agent_id:
                continue
                
            ox, oy = other_agent.position
            distance = math.sqrt((x - ox)**2 + (y - oy)**2)
            if distance <= self.radius:
                nearby.append({
                    "agent_id": other_id,
                    "position": (ox, oy),
                    "distance": distance
                })
                
        # Detect resources
        resources = []
        for resource in agent.pond.resources:
            rx, ry = resource.position
            distance = math.sqrt((x - rx)**2 + (y - ry)**2)
            if distance <= self.radius:
                resources.append({
                    "resource_id": resource.resource_id,
                    "position": (rx, ry),
                    "type": resource.resource_type.value,
                    "distance": distance
                })
                
        return {
            "nearby_agents": nearby,
            "nearby_resources": resources
        }

class PondResource:
    """Resource within the SmallPond environment"""
    def __init__(self, resource_id: str, position: Tuple[int, int], 
                 resource_type: ResourceType = ResourceType.DATA, value: int = 10):
        self.resource_id = resource_id
        self.position = position
        self.resource_type = resource_type
        self.value = value
        self.consumed = False
        
    def consume(self):
        if not self.consumed:
            self.consumed = True
            return self.value
        return 0

class PondAgent:
    """AI Agent within the SmallPond environment"""
    def __init__(self, agent_id: str, pond: 'SmallPond', 
                 position: Tuple[int, int] = (0, 0), energy: int = 100):
        self.agent_id = agent_id
        self.pond = pond
        self.position = position
        self.energy = energy
        self.inbox = asyncio.Queue()
        self.memory = {}
        self.behaviors = []
        self.known_agents = set()
        
    async def send_message(self, recipient_id: str, message: dict):
        """Send a message to another agent in the pond"""
        if recipient_id not in self.pond.agents:
            raise PondError(f"Recipient {recipient_id} not found in pond")
            
        packet = CognitivePacket(
            dest=f"agent:{recipient_id}",
            msg_type=MessageType.POND,
            sender=f"agent:{self.agent_id}",
            group=MessageGroup.POND,
            payload=message
        )
        await self.pond.skc_manager.send_packet(packet)
        
    async def receive_message(self, timeout: float = 1.0) -> Optional[dict]:
        """Receive a message with timeout"""
        try:
            return await asyncio.wait_for(self.inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
            
    async def act(self):
        """Agent decision-making cycle"""
        if self.energy <= 0:
            return "No energy to act"
            
        # Default behavior: Sense surroundings
        sense_action = SenseAction(radius=2)
        result = await sense_action.execute(self)
        self.energy -= sense_action.cost
        
        # Process sensory data
        for agent_info in result.get('nearby_agents', []):
            self.known_agents.add(agent_info['agent_id'])
            
        # Choose a behavior if available
        if self.behaviors:
            behavior = random.choice(self.behaviors)
            return await behavior.execute(self)
            
        # Default random movement
        directions = ["north", "south", "east", "west"]
        move_action = MoveAction(random.choice(directions))
        result = await move_action.execute(self)
        self.energy -= move_action.cost
        return result
        
    def add_behavior(self, behavior: AgentAction):
        """Add a custom behavior to the agent"""
        self.behaviors.append(behavior)
        
    def consume_resource(self, resource: PondResource):
        """Consume a resource to gain energy"""
        if not resource.consumed and self.position == resource.position:
            self.energy += resource.consume()
            return f"Consumed resource {resource.resource_id} (+{resource.value} energy)"
        return "No resource to consume"
        
    def __str__(self):
        return f"Agent({self.agent_id}, pos={self.position}, energy={self.energy})"

class SmallPond:
    """Contained environment for AI agent interaction and learning"""
    def __init__(self, pond_id: str, size: int = 10, max_agents: int = MAX_POND_AGENTS):
        self.pond_id = pond_id
        self.size = size
        self.max_agents = max_agents
        self.skc_manager = SKCManager(node_id=f"pond-{pond_id}")
        self.agents: Dict[str, PondAgent] = {}
        self.resources: List[PondResource] = []
        self._running = False
        self._tick_task = None
        self._lock = asyncio.Lock()
        
        # Register pond message handler
        asyncio.create_task(
            self.skc_manager.register_handler(MessageType.POND, self._handle_pond_message)
        )
        
        # Add route for agent messages
        asyncio.create_task(
            self.skc_manager.router.add_route("agent:*", self.skc_manager.node_id)
        )
        
    async def add_agent(self, agent: PondAgent):
        """Add an agent to the pond"""
        async with self._lock:
            if len(self.agents) >= self.max_agents:
                raise PondError(f"Pond is full (max {self.max_agents} agents)")
                
            if agent.agent_id in self.agents:
                raise PondError(f"Agent {agent.agent_id} already exists")
                
            self.agents[agent.agent_id] = agent
            logger.info(f"Agent {agent.agent_id} added to pond at {agent.position}")
            
    async def remove_agent(self, agent_id: str):
        """Remove an agent from the pond"""
        async with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                logger.info(f"Agent {agent_id} removed from pond")
                
    def add_resource(self, resource: PondResource):
        """Add a resource to the pond"""
        self.resources.append(resource)
        logger.info(f"Resource {resource.resource_id} added at {resource.position}")
        
    def get_random_position(self) -> Tuple[int, int]:
        """Get a random position within the pond"""
        return (random.randint(0, self.size-1), (random.randint(0, self.size-1))
        
    async def start(self):
        """Start the pond simulation"""
        if self._running:
            return
            
        self._running = True
        self._tick_task = asyncio.create_task(self._tick_loop())
        logger.info(f"SmallPond {self.pond_id} started with {len(self.agents)} agents")
        
    async def stop(self):
        """Stop the pond simulation"""
        if not self._running:
            return
            
        self._running = False
        if self._tick_task:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
        logger.info(f"SmallPond {self.pond_id} stopped")
        
    async def _tick_loop(self):
        """Main simulation loop"""
        while self._running:
            start_time = time.time()
            async with self._lock:
                # Process all agents
                for agent_id, agent in list(self.agents.items()):
                    try:
                        result = await agent.act()
                        logger.debug(f"Agent {agent_id} acted: {result}")
                        
                        # Check for resource consumption
                        for resource in self.resources:
                            if agent.position == resource.position and not resource.consumed:
                                consume_result = agent.consume_resource(resource)
                                logger.info(f"Agent {agent_id} {consume_result}")
                                
                    except Exception as e:
                        logger.error(f"Agent {agent_id} error: {e}")
                        
            # Maintain tick rate
            elapsed = time.time() - start_time
            sleep_time = max(0, POND_TICK_INTERVAL - elapsed)
            await asyncio.sleep(sleep_time)
            
    async def _handle_pond_message(self, packet: CognitivePacket):
        """Handle incoming pond messages"""
        try:
            # Message format: agent:<agent_id>
            agent_id = packet.dest.split(':')[1]
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                await agent.inbox.put({
                    "sender": packet.sender.split(':')[1],
                    "payload": packet.payload,
                    "timestamp": packet.timestamp
                })
                return True
            return False
        except Exception as e:
            logger.error(f"Error handling pond message: {e}")
            return False
            
    def status(self) -> dict:
        """Get current pond status"""
        return {
            "pond_id": self.pond_id,
            "size": self.size,
            "agent_count": len(self.agents),
            "resource_count": len(self.resources),
            "running": self._running
        }

# ============================================================================
# Utility Functions
# ============================================================================

def create_cognitive_packet(**kwargs) -> CognitivePacket:
    return CognitivePacket(**kwargs)

def create_skc_resource(**kwargs) -> SKCResource:
    return SKCResource(**kwargs)

# ============================================================================
# Example Agent and Pond Setup
# ============================================================================

class ExplorerAgent(PondAgent):
    """Specialized agent for exploring the pond environment"""
    def __init__(self, agent_id: str, pond: SmallPond):
        super().__init__(agent_id, pond)
        # Add exploration behavior
        self.add_behavior(SenseAction(radius=3))
        self.add_behavior(MoveAction("north"))
        self.add_behavior(MoveAction("south"))
        self.add_behavior(MoveAction("east"))
        self.add_behavior(MoveAction("west"))
        
    async def act(self):
        # Custom exploration logic
        if self.energy <= 0:
            return "No energy to explore"
            
        # Sense surroundings first
        sense_action = SenseAction(radius=3)
        result = await sense_action.execute(self)
        self.energy -= sense_action.cost
        
        # If resources found, move toward nearest
        resources = result.get('nearby_resources', [])
        if resources:
            nearest = min(resources, key=lambda r: r['distance'])
            return await self._move_toward(nearest['position'])
            
        # Random exploration
        return await super().act()
        
    async def _move_toward(self, target: Tuple[int, int]):
        """Move toward a target position"""
        x, y = self.position
        tx, ty = target
        
        # Determine direction
        dx = tx - x
        dy = ty - y
        
        if abs(dx) > abs(dy):
            direction = "east" if dx > 0 else "west"
        else:
            direction = "south" if dy > 0 else "north"
            
        move_action = MoveAction(direction)
        return await move_action.execute(self)

async def demo_pond():
    """Demonstration of the SmallPond environment"""
    # Create pond
    pond = SmallPond("ai-training", size=15)
    
    # Add resources
    for i in range(5):
        position = pond.get_random_position()
        pond.add_resource(PondResource(f"food-{i}", position, value=random.randint(5, 20)))
        
    # Add agents
    agents = []
    for i in range(5):
        agent = ExplorerAgent(f"explorer-{i}", pond)
        agent.position = pond.get_random_position()
        await pond.add_agent(agent)
        agents.append(agent)
        
    # Start pond simulation
    await pond.start()
    
    # Run simulation for 30 seconds
    logger.info("Starting SmallPond simulation...")
    await asyncio.sleep(30)
    
    # Stop and report
    await pond.stop()
    for agent in agents:
        logger.info(f"Agent {agent.agent_id} ended with energy={agent.energy}")
    
    logger.info("SmallPond simulation completed")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Start the demo pond
    asyncio.run(demo_pond())
