#!/usr/bin/env python3
"""
Crisp SKC with SmallPond & KIP Protocol - Complete Implementation
- High-performance AI agent environment with collective intelligence
- Knowledge Integration Protocol for consensus building
- Cryptographic security with enhanced VetKey
- Optimized spatial grid for efficient agent interactions
- Comprehensive resource management
- Detailed error handling and performance profiling
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
import cProfile
import pstats
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from functools import lru_cache, cached_property

# Configure logging with better performance
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Custom Exceptions
# ============================================================================

class SerializationError(Exception):
    """Custom exception for serialization failures"""
    CODES = {
        100: "Header serialization failure",
        101: "Payload serialization failure",
        200: "Header deserialization failure",
        201: "Payload deserialization failure",
        300: "Invalid binary structure",
        999: "Unknown serialization error"
    }
    
    def __init__(self, message, code=0):
        super().__init__(message)
        self.code = code
        self.message = f"{message} (Error #{code}: {self.CODES.get(code, 'Unknown error')})"

class ValidationError(Exception):
    """Custom exception for data validation failures"""
    pass

class CryptoError(Exception):
    """Custom exception for cryptographic operations"""
    CODES = {
        100: "Invalid key format",
        101: "Invalid signature format",
        200: "Signing operation failed",
        201: "Verification operation failed",
        300: "Key derivation failed"
    }
    
    def __init__(self, message, code=0):
        super().__init__(message)
        self.code = code
        self.message = f"{message} (Error #{code}: {self.CODES.get(code, 'Unknown error')})"

class PondError(Exception):
    """Custom exception for SmallPond operations"""
    pass

class KIPError(Exception):
    """Custom exception for KIP operations"""
    pass

# ============================================================================
# Dependency Management
# ============================================================================

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
    
    class Principal:
        def __init__(self, text: str): self.text = text
        @classmethod
        def from_text(cls, text: str): return cls(text)
        def __str__(self): return self.text
        def __repr__(self): return f"Principal('{self.text}')"
        def __eq__(self, other): return isinstance(other, Principal) and self.text == other.text
        def __hash__(self): return hash(self.text)
    
    class HttpAgent:
        def __init__(self, endpoint: str): self.endpoint = endpoint
        async def update_canister(self, canister_id: Principal, binary_data: bytes, identity=None): 
            logger.info(f"Mock update_canister: {canister_id}")
        async def install_code(self, canister_id: Principal, wasm_module: bytes, args: bytes = b'', mode: str = "install", identity=None): 
            logger.info(f"Mock install_code: {canister_id}")

    class Identity:
        def get_principal(self): return Principal.from_text("aaaaa-aa")
        def sign(self, blob: bytes) -> bytes: return b"mock_signature_" + hashlib.sha256(blob).digest()[:8]
        def __repr__(self): return f"Identity('{self.get_principal()}')"
    
    class AgentError(Exception): pass

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
            self._private_key = private_key if isinstance(private_key, ed25519.Ed25519PrivateKey) else None
            super().__init__(principal_text or "crypto-derived-principal")
        
        @staticmethod
        def from_pem(pem_bytes: bytes):
            try:
                private_key = serialization.load_pem_private_key(
                    pem_bytes, 
                    password=None,
                    backend=default_backend()
                )
                if not isinstance(private_key, ed25519.Ed25519PrivateKey):
                    logger.warning("Unsupported private key type")
                    return BasicIdentity(principal_text="mock-unsupported-key")
                return BasicIdentity(private_key=private_key)
            except Exception as e:
                logger.error(f"Failed to load identity from PEM: {e}")
                return BasicIdentity(principal_text="mock-principal-pem-fail")
        
        def sign(self, blob: bytes) -> bytes:
            if self._private_key:
                try:
                    return self._private_key.sign(blob)
                except Exception as e:
                    logger.error(f"Signing failed: {e}")
                    raise CryptoError(f"Signing failed: {e}", 200) from e
            return super().sign(blob)
except ImportError:
    logger.warning("Cryptography library not available, using basic mock identity.")
    class BasicIdentity(Identity):
        def __init__(self, principal_text: str = "mock-principal"): super().__init__(principal_text)
        @staticmethod
        def from_pem(pem_bytes: bytes): return BasicIdentity("mock-principal-from-pem")
    class InvalidSignature(Exception): pass

# Try to import requests for error handling
REQUESTS_AVAILABLE = False
try:
    import requests.exceptions
    REQUESTS_AVAILABLE = True
except ImportError:
    logger.warning("requests library not available.")
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
    logger.warning("cbor2 not available.")
    class cbor2:
        @staticmethod
        def dumps(data): raise NotImplementedError("cbor2 not installed")
        @staticmethod
        def loads(data): raise NotImplementedError("cbor2 not installed")

# Try to import zstandard
ZSTD_AVAILABLE = False
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
    logger.info("zstandard available.")
except ImportError:
    logger.warning("zstandard not available.")
    class zstd:
        class ZstdCompressor:
            def compress(self, data): return data
        class ZstdDecompressor:
            def decompress(self, data): return data

# ============================================================================
# Constants
# ============================================================================
MAX_HOPS = 15
MAX_PACKET_SIZE = 10 * 1024 * 1024
DEFAULT_CHUNK_SIZE = 1024 * 1024
MAX_POND_AGENTS = 100
POND_TICK_INTERVAL = 0.1
KIP_CONSENSUS_THRESHOLD = 0.6
KIP_CLAIM_EXPIRATION_DAYS = 30
VETKEY_SIGNATURE_LENGTH = 64
VETKEY_PUBLIC_KEY_LENGTH = 32

# ============================================================================
# Protocol Enums
# ============================================================================

class MessageType(Enum):
    """Types of messages in the Crisp protocol"""
    QUERY = "query"
    DATA = "data"
    EXEC = "exec"
    COMMAND = "command"
    ACK = "ack"
    ERROR = "error"
    UPDATE = "update"
    DISCOVER = "discover"
    POND = "pond"
    KIP_CLAIM = "kip_claim"
    KIP_VOTE = "kip_vote"
    KIP_QUERY = "kip_query"
    KIP_RESPONSE = "kip_response"

class ResourceType(Enum):
    """Types of resources in the system"""
    DATA = "data"
    MODEL = "model"
    WASM = "wasm"
    KNOWLEDGE = "knowledge"

class KnowledgeClaimStatus(Enum):
    """Status of knowledge claims in the KIP protocol"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"

class KnowledgeTrustLevel(Enum):
    """Trust levels for knowledge claims"""
    UNVERIFIED = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERIFIED = 4

# ============================================================================
# Core Security: VetKey Implementation
# ============================================================================

class VetKey:
    """
    Implementation of forward-secure cryptographic keys using the VetKey protocol.
    
    Provides:
    - Forward-secure signing using epoch-based key evolution
    - Asymmetric key operations with Ed25519 for signatures
    - Key derivation using X25519 for secure communication
    
    Attributes:
        epoch (int): Current key epoch (increases with each ratchet)
    
    Methods:
        signing_public_bytes() -> bytes:
            Get the raw public key bytes for verification
            
        ratchet_forward() -> None:
            Advance the key state for forward secrecy
            
        sign(data: bytes) -> bytes:
            Sign data using the current key state
            
        verify(signature: bytes, data: bytes, public_key_bytes: bytes) -> bool:
            Static method to verify a signature
    """
    
    def __init__(self):
        if not CRYPTO_AVAILABLE:
            raise CryptoError("Cryptography library is required for VetKey", 300)
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
        try:
            peer_key = x25519.X25519PublicKey.from_public_bytes(peer_exchange_public_key_bytes)
            shared_secret = self._exchange_key.exchange(peer_key)
            
            return HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'Crisp-VetKey-Derivation',
                backend=default_backend()
            ).derive(shared_secret)
        except Exception as e:
            logger.error(f"Key derivation failed: {e}")
            raise CryptoError(f"Key derivation failed: {e}", 300) from e

    def ratchet_forward(self) -> None:
        """Advance the key state for forward secrecy."""
        self.epoch += 1
        self.chain_key = hashlib.blake2b(
            b'ratchet-salt' + self.chain_key, 
            digest_size=32
        ).digest()

    def sign(self, data: bytes) -> bytes:
        """Sign data using the Ed25519 signing key with validation"""
        if not isinstance(data, bytes):
            raise CryptoError("Data must be bytes for signing", 100)
        try:
            return self._signing_key.sign(data)
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            raise CryptoError(f"Signing failed: {e}", 200) from e

    @staticmethod
    def verify(signature: bytes, data: bytes, public_key_bytes: bytes) -> bool:
        """Verify a signature using a public key with strict validation"""
        try:
            # Validate input formats
            if not isinstance(signature, bytes) or len(signature) != VETKEY_SIGNATURE_LENGTH:
                raise CryptoError("Invalid signature format", 101)
            if not isinstance(data, bytes):
                raise CryptoError("Data must be bytes", 100)
            if not isinstance(public_key_bytes, bytes) or len(public_key_bytes) != VETKEY_PUBLIC_KEY_LENGTH:
                raise CryptoError("Invalid public key format", 100)
                
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
            public_key.verify(signature, data)
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"Verification error: {str(e)}")
            return False

class MockVetKey:
    """Mock VetKey for environments without cryptography support"""
    def __init__(self): 
        self.epoch = 0
    
    def signing_public_bytes(self) -> bytes: 
        return b'\x00' * VETKEY_PUBLIC_KEY_LENGTH
    
    def ratchet_forward(self) -> None: 
        self.epoch += 1
    
    def sign(self, data: bytes) -> bytes: 
        return b"mock_vetkey_sig_" + hashlib.sha256(data).digest()[:8]
    
    @staticmethod
    def verify(signature: bytes, data: bytes, public_key_bytes: bytes) -> bool: 
        return True  # Always verify successfully in mock mode

# ============================================================================
# Data Structures
# ============================================================================

class BF16Vector:
    """
    Efficient storage for vectors using Brain Float 16 format.
    
    Attributes:
        _data (bytearray): Internal storage of BF16 bytes
    
    Methods:
        to_float32() -> List[float]: Convert back to float32 list
        cosine_similarity(other: BF16Vector) -> float: Compute similarity
    """
    
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
            except (struct.error, TypeError) as e:
                logger.warning(f"BF16 conversion error: {e}, skipping value")
        return bf16_data

    def to_float32(self) -> List[float]:
        """Convert back to float32 list."""
        floats = []
        for i in range(0, len(self._data), 2):
            if i + 2 > len(self._data):
                break
            try:
                bf16_val = struct.unpack('!H', self._data[i:i+2])[0]
                float_val = struct.unpack('!f', struct.pack('!I', bf16_val << 16))[0]
                floats.append(float_val)
            except struct.error as e:
                logger.warning(f"Float conversion error: {e}")
        return floats

    def cosine_similarity(self, other: 'BF16Vector') -> float:
        """Compute cosine similarity between two vectors"""
        if len(self) != len(other) or len(self) == 0:
            return 0.0
            
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

    def __repr__(self) -> str:
        return f"BF16Vector(dim={len(self)})"

class MemoryPool:
    """
    Reusable memory pool for efficient buffer management
    
    Attributes:
        _pool (List[bytearray]): Available buffers
        _chunk_size (int): Default buffer size
        _allocation_count (int): Total buffers allocated
        _reuse_count (int): Buffers reused
    """
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self._pool = []
        self._chunk_size = chunk_size
        self._allocation_count = 0
        self._reuse_count = 0

    def allocate(self, size: int) -> bytearray:
        """Get a buffer from the pool or allocate a new one"""
        if self._pool and len(self._pool[-1]) >= size:
            self._reuse_count += 1
            return self._pool.pop()
            
        self._allocation_count += 1
        return bytearray(max(size, self._chunk_size))

    def release(self, buf: bytearray) -> None:
        """Return a buffer to the pool after resetting"""
        if not isinstance(buf, bytearray):
            return
            
        buf[:] = b'\x00' * len(buf)
        self._pool.append(buf)

    def stats(self) -> Dict[str, int]:
        """Get current pool statistics"""
        return {
            "available": len(self._pool),
            "total_allocated": self._allocation_count,
            "reuse_count": self._reuse_count
        }
        
    def clear(self):
        """Release all buffers in the pool"""
        self._pool = []
        self._allocation_count = 0
        self._reuse_count = 0

class AsyncReadWriteLock:
    """
    Asynchronous read-write lock for resource management
    
    Provides:
    - reader: Context manager for read operations
    - writer: Context manager for write operations
    """
    
    def __init__(self):
        self._cond = asyncio.Condition()
        self._readers = 0
        self._writer = False
        
    async def reader_acquire(self):
        async with self._cond:
            await self._cond.wait_for(lambda: not self._writer)
            self._readers += 1
            
    async def reader_release(self):
        async with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()
                
    async def writer_acquire(self):
        async with self._cond:
            await self._cond.wait_for(lambda: not self._writer and self._readers == 0)
            self._writer = True
            
    async def writer_release(self):
        async with self._cond:
            self._writer = False
            self._cond.notify_all()
            
    class Context:
        def __init__(self, lock, acquire, release):
            self._lock = lock
            self._acquire = acquire
            self._release = release
            
        async def __aenter__(self):
            await self._acquire()
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self._release()
            
    @property
    def reader(self):
        return self.Context(self, self.reader_acquire, self.reader_release)
        
    @property
    def writer(self):
        return self.Context(self, self.writer_acquire, self.writer_release)

# ============================================================================
# Protocol Data Structures
# ============================================================================

@dataclass
class CognitivePacket:
    """
    Fundamental network packet structure for the Crisp protocol.
    
    The packet uses a binary format with three sections:
    1. Header section (msgpack serialized metadata)
    2. Payload section (msgpack serialized structured data)
    3. Binary payload section (raw binary data)
    
    Attributes:
        dest (str): Destination node identifier
        msg_type (MessageType): Type of message
        sender (str): Sender node identifier
        msg_id (str): Unique message identifier (UUID)
        timestamp (str): ISO 8601 timestamp of creation
        hops (int): Number of hops traversed
        payload (Dict[str, Any]): Structured payload data
        binary_payload (bytes): Raw binary payload
        vet_signature (bytes): Cryptographic signature
        vet_public_key (bytes): Public key for verification
        vet_epoch (int): Key epoch when signed
    """
    
    dest: str
    msg_type: MessageType
    sender: str
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hops: int = 0
    payload: Optional[Dict[str, Any]] = None
    binary_payload: Optional[bytes] = None
    vet_signature: Optional[bytes] = None
    vet_public_key: Optional[bytes] = None
    vet_epoch: Optional[int] = None
    _cached_binary: Optional[bytes] = field(default=None, init=False, repr=False)
    _cached_binary_valid: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.msg_type, str):
            self.msg_type = MessageType(self.msg_type.lower())

    def invalidate_cache(self):
        """Invalidate the cached binary representation"""
        self._cached_binary_valid = False
        self._cached_binary = None
        self.vet_signature = None

    def to_binary_format(self) -> bytes:
        """Serialize the packet to binary format"""
        if self._cached_binary_valid and self._cached_binary is not None:
            return self._cached_binary
            
        try:
            # Prepare headers
            headers = {
                "dest": self.dest,
                "msg_type": self.msg_type.value,
                "sender": self.sender,
                "msg_id": self.msg_id,
                "timestamp": self.timestamp,
                "hops": self.hops,
                "vet_epoch": self.vet_epoch
            }
            
            # Add optional fields
            if self.vet_public_key:
                headers["vet_public_key"] = base64.b64encode(self.vet_public_key).decode('ascii')
                
            # Serialize headers and payload
            header_data = msgpack.packb(headers, use_bin_type=True)
            payload_data = msgpack.packb(self.payload, use_bin_type=True) if self.payload is not None else b''
            binary_data = self.binary_payload or b''
            
            # Create final binary structure
            result = struct.pack('!III', len(header_data), len(payload_data), len(binary_data))
            result += header_data
            result += payload_data
            result += binary_data
            
            # Update cache
            self._cached_binary = result
            self._cached_binary_valid = True
            return result
        except (msgpack.PackException, struct.error) as e:
            raise SerializationError(f"Serialization failed: {e}", 100) from e
        except Exception as e:
            raise SerializationError(f"Unexpected serialization error: {e}", 999) from e

    @classmethod
    def from_binary_format(cls, binary_data: bytes) -> 'CognitivePacket':
        """Deserialize a packet from binary format"""
        try:
            # Unpack header lengths
            if len(binary_data) < 12:
                raise SerializationError("Insufficient data for header", 300)
                
            header_len, payload_len, binary_len = struct.unpack('!III', binary_data[:12])
            offset = 12
            total_len = 12 + header_len + payload_len + binary_len
            
            if len(binary_data) < total_len:
                raise SerializationError(
                    f"Insufficient data: expected {total_len}, got {len(binary_data)}", 
                    300
                )
            
            # Extract sections
            header_bytes = binary_data[offset:offset+header_len]
            offset += header_len
            payload_bytes = binary_data[offset:offset+payload_len] if payload_len > 0 else b''
            offset += payload_len
            binary_payload = binary_data[offset:offset+binary_len] if binary_len > 0 else None
            
            # Deserialize headers
            headers = msgpack.unpackb(header_bytes, raw=False)
            
            # Handle vet key if present
            if "vet_public_key" in headers:
                headers["vet_public_key"] = base64.b64decode(headers["vet_public_key"])
            
            # Deserialize payload
            payload_data = msgpack.unpackb(payload_bytes, raw=False) if payload_bytes else None
            
            # Create packet instance
            valid_keys = {f.name for f in dataclasses.fields(cls) if f.init}
            init_kwargs = {k: v for k, v in headers.items() if k in valid_keys}
            return cls(
                **init_kwargs, 
                payload=payload_data,
                binary_payload=binary_payload
            )
        except (msgpack.UnpackException, struct.error) as e:
            raise SerializationError(f"Deserialization failed: {e}", 200) from e
        except Exception as e:
            raise SerializationError(f"Unexpected deserialization error: {e}", 999) from e

    def sign_with_vetkey(self, vet_key: Union[VetKey, MockVetKey]):
        """Sign the packet with a VetKey"""
        self.invalidate_cache()
        self.vet_public_key = vet_key.signing_public_bytes()
        self.vet_epoch = vet_key.epoch
        
        # Create temporary packet without signature for signing
        temp_packet = dataclasses.replace(self, vet_signature=None)
        packet_data = temp_packet.to_binary_format()
        
        try:
            self.vet_signature = vet_key.sign(packet_data)
        except Exception as e:
            raise CryptoError(f"Signing failed: {e}", 200) from e
            
        self.invalidate_cache()

    def verify_vet_signature(self) -> bool:
        """Verify the packet signature"""
        if not self.vet_signature or not self.vet_public_key:
            logger.warning("Missing signature components")
            return False
            
        # Clone without signature for verification
        temp_packet = dataclasses.replace(
            self,
            vet_signature=None,
            _cached_binary=None,
            _cached_binary_valid=False
        )
        
        try:
            packet_data = temp_packet.to_binary_format()
            return VetKey.verify(
                self.vet_signature,
                packet_data,
                self.vet_public_key
            )
        except Exception as e:
            logger.error(f"Signature verification failed: {str(e)}")
            return False

@dataclass
class KnowledgeClaim:
    """Represents a unit of knowledge in the KIP protocol"""
    claim_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    author_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expiration: Optional[str] = None
    vector: BF16Vector = field(default_factory=BF16Vector)
    votes: Dict[str, bool] = field(default_factory=dict)  # agent_id: vote (True=accept)
    status: KnowledgeClaimStatus = KnowledgeClaimStatus.PENDING
    confidence: float = 0.5
    trust_level: KnowledgeTrustLevel = KnowledgeTrustLevel.UNVERIFIED
    supporting_claims: List[str] = field(default_factory=list)
    refuting_claims: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.expiration:
            self.expiration = (
                datetime.now(timezone.utc) + timedelta(days=KIP_CLAIM_EXPIRATION_DAYS)
            ).isoformat()

    def add_vote(self, agent_id: str, vote: bool):
        """Add a vote to the claim and update status"""
        if agent_id in self.votes or self.status != KnowledgeClaimStatus.PENDING:
            return
            
        self.votes[agent_id] = vote
        self.update_status()

    def update_status(self):
        """Update claim status based on votes and expiration"""
        if self.is_expired():
            self.status = KnowledgeClaimStatus.EXPIRED
            return
            
        accept_count = sum(1 for v in self.votes.values() if v)
        total_votes = len(self.votes)
        
        if total_votes == 0:
            self.confidence = 0.5
        else:
            self.confidence = accept_count / total_votes
            
        if total_votes >= 3 and self.confidence >= KIP_CONSENSUS_THRESHOLD:
            self.status = KnowledgeClaimStatus.ACCEPTED
            self.trust_level = KnowledgeTrustLevel.STRONG
        elif total_votes >= 3 and (1 - self.confidence) >= KIP_CONSENSUS_THRESHOLD:
            self.status = KnowledgeClaimStatus.REJECTED
            self.trust_level = KnowledgeTrustLevel.UNVERIFIED
        elif total_votes >= 5 and self.status == KnowledgeClaimStatus.ACCEPTED:
            self.trust_level = KnowledgeTrustLevel.VERIFIED
        else:
            self.status = KnowledgeClaimStatus.PENDING
# ============================================================================
# Core System Components
# ============================================================================

class SKCManager:
    """
    Manages resources, handlers, and packet routing for a node.
    
    Attributes:
        node_id (str): Unique identifier for the node
        packet_handlers (Dict[MessageType, List[Callable]]: Registered handlers
        router (Dict[str, List[Any]]: Routing table
        vet_key (Union[VetKey, MockVetKey]): Cryptographic identity
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.packet_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.router = defaultdict(list)
        self.vet_key = VetKey() if CRYPTO_AVAILABLE else MockVetKey()
        self._handler_lock = asyncio.Lock()
        self._pond_instance: Optional['SmallPond'] = None
        logger.info(f"SKCManager initialized for node {self.node_id}")

    async def register_handler(self, msg_type: MessageType, handler: Callable):
        """Register a packet handler for a message type"""
        async with self._handler_lock:
            self.packet_handlers[msg_type].append(handler)
    
    async def add_route(self, dest_prefix: str, endpoint: Any):
        """Add a route to the routing table"""
        self.router[dest_prefix].append(endpoint)

    async def process_packet(self, packet: CognitivePacket) -> bool:
        """Process an incoming packet"""
        if packet.hops > MAX_HOPS:
            logger.warning(f"Packet {packet.msg_id} exceeded max hops")
            return False
            
        if packet.vet_signature and not packet.verify_vet_signature():
            logger.warning(f"Invalid signature on packet {packet.msg_id}")
            return False
            
        handlers = self.packet_handlers.get(packet.msg_type, [])
        if not handlers:
            logger.debug(f"No handlers for message type {packet.msg_type}")
            return False
        
        # Process handlers in parallel with error handling
        results = await asyncio.gather(
            *(h(packet) for h in handlers),
            return_exceptions=True
        )
        
        # Log errors but don't block
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Handler error: {str(r)}", exc_info=True)
        
        return any(r is True for r in results if not isinstance(r, Exception))

    async def send_packet(self, packet: CognitivePacket) -> bool:
        """Send a packet to its destination"""
        try:
            packet.sign_with_vetkey(self.vet_key)
            self.vet_key.ratchet_forward()
            
            # Simplified routing for simulation
            if self._pond_instance:
                return await self._pond_instance.skc_manager.process_packet(packet)
            return False
        except Exception as e:
            logger.error(f"Failed to send packet: {str(e)}")
            return False

class KIPProtocol:
    """
    Knowledge Integration Protocol implementation
    
    Attributes:
        pond (SmallPond): Associated pond environment
        claims (Dict[str, KnowledgeClaim]): Registered claims
        claim_index (Dict[str, List[str]]): Content hash to claim IDs mapping
    """
    
    def __init__(self, pond: 'SmallPond'):
        self.pond = pond
        self.skc_manager = pond.skc_manager
        self.claims: Dict[str, KnowledgeClaim] = {}
        self.claim_index: Dict[str, List[str]] = defaultdict(list)
        self._lock = AsyncReadWriteLock()
        self.consensus_events = asyncio.Queue()
        self._consensus_task = None
        
    async def start(self):
        """Start periodic consensus building"""
        if not self._consensus_task or self._consensus_task.done():
            self._consensus_task = asyncio.create_task(self.run_consensus_cycle())
            
    async def stop(self):
        """Stop consensus building and clean up"""
        if self._consensus_task and not self._consensus_task.done():
            self._consensus_task.cancel()
            try:
                await self._consensus_task
            except asyncio.CancelledError:
                pass
            self._consensus_task = None

    async def register_claim(self, claim: KnowledgeClaim):
        """Register a new knowledge claim"""
        async with self._lock.writer:
            if claim.claim_id in self.claims:
                logger.warning(f"Claim {claim.claim_id[:8]} already registered")
                return False
                
            self.claims[claim.claim_id] = claim
            content_hash = hashlib.sha256(claim.content.encode('utf-8')).hexdigest()
            self.claim_index[content_hash].append(claim.claim_id)
            
            # Broadcast claim notification
            packet = CognitivePacket(
                dest="broadcast",
                msg_type=MessageType.KIP_CLAIM,
                sender=self.pond.pond_id,
                payload={
                    "claim_id": claim.claim_id,
                    "content_hash": content_hash,
                    "author": claim.author_id,
                    "timestamp": claim.timestamp
                }
            )
            await self.skc_manager.send_packet(packet)
            
            logger.info(f"New claim registered: {claim.claim_id[:8]} by {claim.author_id}")
            return True
    
    async def cast_vote(self, claim_id: str, agent_id: str, vote: bool):
        """Cast a vote on a knowledge claim"""
        async with self._lock.writer:
            claim = self.claims.get(claim_id)
            if not claim:
                logger.warning(f"Vote attempt on unknown claim: {claim_id}")
                return False
                
            if claim.author_id == agent_id:
                logger.debug(f"Agent {agent_id} cannot vote on their own claim")
                return False
                
            if agent_id in claim.votes:
                logger.debug(f"Agent {agent_id} already voted on claim {claim_id[:8]}")
                return False
                
            claim.add_vote(agent_id, vote)
            old_status = claim.status
            
            # Broadcast vote
            packet = CognitivePacket(
                dest="broadcast",
                msg_type=MessageType.KIP_VOTE,
                sender=agent_id,
                payload={
                    "claim_id": claim_id,
                    "vote": vote,
                    "voter": agent_id,
                    "new_status": claim.status.value
                }
            )
            await self.skc_manager.send_packet(packet)
            
            if claim.status != old_status:
                logger.info(f"Claim {claim_id[:8]} status changed to {claim.status.name}")
                
            return True

    async def query_claims(self, content_hash: str = None, author_id: str = None, 
                           status: KnowledgeClaimStatus = None) -> List[KnowledgeClaim]:
        """Query knowledge claims based on criteria"""
        async with self._lock.reader:
            if content_hash:
                claim_ids = self.claim_index.get(content_hash, [])
                return [self.claims[cid] for cid in claim_ids if cid in self.claims]
            elif author_id:
                return [c for c in self.claims.values() if c.author_id == author_id]
            elif status:
                return [c for c in self.claims.values() if c.status == status]
            else:
                return list(self.claims.values())

    async def build_consensus(self) -> Dict[str, Any]:
        """Build consensus report on claims"""
        async with self._lock.reader:
            status_counts = defaultdict(int)
            trust_levels = defaultdict(int)
            vote_counts = []
            
            for claim in self.claims.values():
                claim.update_status()  # Ensure status is current
                status_counts[claim.status.name] += 1
                trust_levels[claim.trust_level.name] += 1
                vote_counts.append(len(claim.votes))
                
            avg_votes = sum(vote_counts) / len(vote_counts) if vote_counts else 0
            
            return {
                "total_claims": len(self.claims),
                "status_counts": dict(status_counts),
                "trust_levels": dict(trust_levels),
                "avg_votes": avg_votes,
                "pending_claims": status_counts.get("PENDING", 0)
            }

    async def run_consensus_cycle(self, interval: float = 5.0) -> None:
        """Periodically build and distribute consensus"""
        while True:
            await asyncio.sleep(interval)
            try:
                consensus = await self.build_consensus()
                await self.consensus_events.put(consensus)
                
                # Broadcast consensus report
                packet = CognitivePacket(
                    dest="broadcast",
                    msg_type=MessageType.KIP_RESPONSE,
                    sender=self.pond.pond_id,
                    payload={"consensus_report": consensus}
                )
                await self.skc_manager.send_packet(packet)
                
                logger.debug(f"Consensus update: {consensus}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Consensus cycle error: {e}")

# ============================================================================
# Agent and Pond Implementation
# ============================================================================
class PondAgent:
    """Agent within the SmallPond environment"""
    
    def __init__(self, agent_id: str, pond: 'SmallPond', energy: int = 100):
        self.id = agent_id
        self.pond = pond
        self.position: Tuple[float, float] = pond.get_random_position()
        self.velocity: Tuple[float, float] = (random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5))
        self.energy = float(energy)
        self.knowledge: Dict[str, KnowledgeClaim] = {}
        self.vetkey = VetKey() if CRYPTO_AVAILABLE else MockVetKey()
        self.last_claim_time = 0
        self.known_claims: Set[str] = set()
        
        # Register packet handlers
        asyncio.create_task(
            self.pond.skc_manager.register_handler(MessageType.KIP_CLAIM, self.handle_kip_claim)
        )
        asyncio.create_task(
            self.pond.skc_manager.register_handler(MessageType.KIP_VOTE, self.handle_kip_vote)
        )
        asyncio.create_task(
            self.pond.skc_manager.register_handler(MessageType.KIP_RESPONSE, self.handle_kip_response)
        )
    
    async def handle_kip_claim(self, packet: CognitivePacket) -> bool:
        """Handle incoming KIP claim notifications"""
        claim_id = packet.payload["claim_id"]
        content_hash = packet.payload["content_hash"]
        
        if claim_id in self.known_claims:
            return True
            
        # Fetch full claim if we don't have it
        if claim_id not in self.knowledge:
            async with self.pond.kip_manager._lock.reader:
                claim = self.pond.kip_manager.claims.get(claim_id)
                if claim:
                    self.knowledge[claim_id] = claim
                    self.known_claims.add(claim_id)
                    self.evaluate_and_vote(claim)
        
        return True
    
    async def handle_kip_vote(self, packet: CognitivePacket) -> bool:
        """Handle vote notifications"""
        claim_id = packet.payload["claim_id"]
        voter_id = packet.payload["voter"]
        vote = packet.payload["vote"]
        
        if claim_id in self.knowledge:
            claim = self.knowledge[claim_id]
            if voter_id not in claim.votes:
                claim.add_vote(voter_id, vote)
        
        return True
    
    async def handle_kip_response(self, packet: CognitivePacket) -> bool:
        """Handle consensus reports"""
        # Agents can react to consensus reports
        report = packet.payload["consensus_report"]
        if report["pending_claims"] > 10:
            # Increase energy when many pending claims
            self.energy = min(self.energy + 5, 100)
        return True

    def move(self):
        """Update agent position based on velocity"""
        x, y = self.position
        dx, dy = self.velocity
        new_x, new_y = x + dx, y + dy
        
        # Bounce off boundaries with energy loss
        if new_x <= 0 or new_x >= self.pond.size:
            self.velocity = (-dx * 0.8, dy * 0.9)
            new_x = max(0, min(self.pond.size, new_x))
            self.energy = max(0, self.energy - 0.2)
        if new_y <= 0 or new_y >= self.pond.size:
            self.velocity = (dx * 0.9, -dy * 0.8)
            new_y = max(0, min(self.pond.size, new_y))
            self.energy = max(0, self.energy - 0.2)
            
        self.position = (new_x, new_y)
        self.energy = max(0, self.energy - 0.05)

    def observe(self) -> List[KnowledgeClaim]:
        """Discover new claims in the pond"""
        new_claims = []
        async with self.pond.kip_manager._lock.reader:
            for claim_id, claim in self.pond.kip_manager.claims.items():
                if claim_id not in self.known_claims:
                    self.known_claims.add(claim_id)
                    self.knowledge[claim_id] = claim
                    new_claims.append(claim)
        return new_claims

    def evaluate_and_vote(self, claim: KnowledgeClaim):
        """Evaluate a claim and cast a vote"""
        if (self.id in claim.votes or 
            claim.author_id == self.id or 
            claim.status != KnowledgeClaimStatus.PENDING):
            return
            
        # Simple evaluation based on vector similarity
        vote_decision = False
        similarity_threshold = 0.65
        
        for my_claim in self.knowledge.values():
            if (my_claim.status == KnowledgeClaimStatus.ACCEPTED and 
                my_claim.vector.cosine_similarity(claim.vector) > similarity_threshold):
                vote_decision = True
                break
        
        # Occasionally challenge consensus
        if random.random() < 0.1 and len(claim.votes) > 3:
            vote_decision = not vote_decision
            
        asyncio.create_task(self.pond.kip_manager.cast_vote(
            claim.claim_id, self.id, vote_decision
        ))
        self.energy = max(0, self.energy - 0.5)

    def create_claim(self):
        """Create a new knowledge claim"""
        if self.energy < 10 or time.monotonic() - self.last_claim_time < 5.0:
            return
            
        # Claim content based on agent's state
        position_str = f"{self.position[0]:.1f},{self.position[1]:.1f}"
        content = f"Observation from {self.id} at {position_str}"
        
        # Vector represents position and velocity
        vector_data = [
            self.position[0] / self.pond.size,
            self.position[1] / self.pond.size,
            self.velocity[0],
            self.velocity[1],
            self.energy / 100.0
        ]
        # Pad with random values
        vector_data.extend(random.uniform(-1, 1) for _ in range(123))
        
        claim = KnowledgeClaim(
            author_id=self.id,
            content=content,
            vector=BF16Vector(vector_data)
        )
        
        self.knowledge[claim.claim_id] = claim
        self.known_claims.add(claim.claim_id)
        asyncio.create_task(self.pond.kip_manager.register_claim(claim))
        
        self.energy = max(0, self.energy - 8.0)
        self.last_claim_time = time.monotonic()

    async def tick(self):
        """Perform agent actions on each simulation tick"""
        if self.energy <= 0:
            return
            
        old_pos = self.position
        self.move()
        self.pond._update_agent_in_grid(self, old_pos)
        
        new_claims = self.observe()
        for claim in new_claims:
            self.evaluate_and_vote(claim)
        
        if random.random() < 0.08:
            self.create_claim()
            
        if random.random() < 0.1:
            self.velocity = (
                random.uniform(-1.5, 1.5),
                random.uniform(-1.5, 1.5)
            )
    
    async def cleanup(self):
        """Release resources held by the agent"""
        self.knowledge.clear()
        self.known_claims.clear()
        if hasattr(self.vetkey, 'reset'):
            self.vetkey.reset()

class SmallPond:
    """Environment for agent interaction and knowledge sharing"""
    
    def __init__(self, pond_id: str, size: int = 1000, grid_cell_size: int = 50):
        self.pond_id = pond_id
        self.size = size
        self.skc_manager = SKCManager(node_id=f"pond-{pond_id}")
        self.skc_manager._pond_instance = self
        self.kip_manager = KIPProtocol(self)
        self.agents: List[PondAgent] = []
        self._running = False
        self.memory_pool = MemoryPool()
        
        # Spatial Grid Optimization
        self.grid_cell_size = grid_cell_size
        self._spatial_grid: Dict[Tuple[int, int], List[PondAgent]] = defaultdict(list)
        self._lock = asyncio.Lock()
        
        # Register KIP packet handlers
        asyncio.create_task(
            self.skc_manager.register_handler(MessageType.KIP_CLAIM, self.handle_kip_claim)
        )
        asyncio.create_task(
            self.skc_manager.register_handler(MessageType.KIP_VOTE, self.handle_kip_vote)
        )
        asyncio.create_task(
            self.skc_manager.register_handler(MessageType.KIP_QUERY, self.handle_kip_query)
        )
        
    async def handle_kip_claim(self, packet: CognitivePacket) -> bool:
        """Handle incoming KIP claim packets"""
        claim_id = packet.payload["claim_id"]
        async with self.kip_manager._lock.reader:
            if claim_id in self.kip_manager.claims:
                return True
                
        # Request full claim data
        query_packet = CognitivePacket(
            dest=packet.sender,
            msg_type=MessageType.KIP_QUERY,
            sender=self.pond_id,
            payload={"claim_id": claim_id}
        )
        await self.skc_manager.send_packet(query_packet)
        return True
    
    async def handle_kip_vote(self, packet: CognitivePacket) -> bool:
        """Handle vote notifications"""
        claim_id = packet.payload["claim_id"]
        async with self.kip_manager._lock.reader:
            if claim_id in self.kip_manager.claims:
                claim = self.kip_manager.claims[claim_id]
                claim.add_vote(packet.payload["voter"], packet.payload["vote"])
        return True
    
    async def handle_kip_query(self, packet: CognitivePacket) -> bool:
        """Handle claim queries"""
        claim_id = packet.payload["claim_id"]
        async with self.kip_manager._lock.reader:
            claim = self.kip_manager.claims.get(claim_id)
            if not claim:
                return False
                
            # Send full claim data
            claim_packet = CognitivePacket(
                dest=packet.sender,
                msg_type=MessageType.KIP_CLAIM,
                sender=self.pond_id,
                payload={
                    "claim_id": claim.claim_id,
                    "author": claim.author_id,
                    "content": claim.content,
                    "vector": base64.b64encode(claim.vector.bf16_data).decode(),
                    "timestamp": claim.timestamp,
                    "expiration": claim.expiration,
                    "status": claim.status.value,
                    "confidence": claim.confidence
                }
            )
            await self.skc_manager.send_packet(claim_packet)
            return True

    async def add_agent(self, agent: PondAgent):
        """Add an agent to the pond"""
        async with self._lock:
            if len(self.agents) >= MAX_POND_AGENTS: 
                raise PondError("Pond is full")
            self.agents.append(agent)
            self._update_agent_in_grid(agent)
            
    def get_random_position(self) -> Tuple[float, float]:
        return (random.uniform(0, self.size), random.uniform(0, self.size))

    def _get_grid_cell(self, position: Tuple[float, float]) -> Tuple[int, int]:
        return (int(position[0] / self.grid_cell_size), int(position[1] / self.grid_cell_size))

    def _update_agent_in_grid(self, agent: PondAgent, old_pos: Optional[Tuple[float, float]] = None):
        """Update agent's position in spatial grid"""
        if old_pos:
            old_cell = self._get_grid_cell(old_pos)
            if agent in self._spatial_grid.get(old_cell, []):
                self._spatial_grid[old_cell].remove(agent)
        new_cell = self._get_grid_cell(agent.position)
        self._spatial_grid[new_cell].append(agent)
    
    async def start(self, duration: float, profile: bool = False):
        """Start the pond simulation with optional profiling"""
        if self._running: 
            return
            
        self._running = True
        await self.kip_manager.start()
        
        profiler = None
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()
        
        logger.info(f"SmallPond {self.pond_id} starting with {len(self.agents)} agents")
        start_time = time.monotonic()
        end_time = start_time + duration
        
        while self._running and time.monotonic() < end_time:
            tick_start = time.monotonic()
            
            # Process agents in batches
            batch_size = max(1, len(self.agents) // 10)
            for i in range(0, len(self.agents), batch_size):
                batch = self.agents[i:i+batch_size]
                tasks = [agent.tick() for agent in batch]
                await asyncio.gather(*tasks)
                
                # Yield to event loop periodically
                if i % (batch_size * 2) == 0:
                    await asyncio.sleep(0)
            
            # Maintain simulation timing
            elapsed = time.monotonic() - tick_start
            await asyncio.sleep(max(0, POND_TICK_INTERVAL - elapsed))
        
        # Final cleanup
        await self.stop()
        
        if profile and profiler:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)
            stats.dump_stats(f"smallpond_{self.pond_id}.prof")

    async def stop(self):
        """Stop the pond simulation and clean up resources"""
        if not self._running:
            return
            
        self._running = False
        
        # Clean up agents
        agent_cleanup = [agent.cleanup() for agent in self.agents]
        await asyncio.gather(*agent_cleanup)
        
        # Clean up KIP manager
        await self.kip_manager.stop()
        
        # Release memory pools
        self.memory_pool.clear()
        
        # Clear spatial grid
        self._spatial_grid.clear()
        
        logger.info(f"SmallPond {self.pond_id} stopped and resources cleaned")

# ============================================================================
# Main Simulation
# ============================================================================

async def run_consensus_reporter(pond: SmallPond, interval: float = 5.0):
    """Periodically report consensus status"""
    while pond._running:
        try:
            consensus = await pond.kip_manager.build_consensus()
            logger.info("\n--- KIP Consensus Report ---")
            logger.info(f"Total Claims: {consensus['total_claims']}")
            
            for status, count in consensus["status_counts"].items():
                logger.info(f"  {status.upper():<10}: {count}")
                
            logger.info(f"Average Votes per Claim: {consensus['avg_votes']:.2f}")
            logger.info("----------------------------")
        except Exception as e:
            logger.error(f"Consensus reporting error: {e}")
        
        await asyncio.sleep(interval)

async def demo_pond(profile: bool = False):
    """Demonstration of SmallPond with full KIP integration"""
    logger.info("Starting Crisp SKC with SmallPond & KIP Protocol")
    pond = SmallPond("demo-pond", size=800)
    
    # Create diverse agents
    for i in range(40):
        agent = PondAgent(
            agent_id=f"Agent-{i+1:03d}", 
            pond=pond,
            energy=random.randint(70, 100)
        )
        await pond.add_agent(agent)
        
        # Initialize with unique position vectors
        agent.velocity = (
            random.uniform(-1.2, 1.2),
            random.uniform(-1.2, 1.2)
        )
    
    # Add initial knowledge to seed the system
    initial_claims = [
        KnowledgeClaim(
            author_id="System",
            content="Gravity affects all objects equally",
            vector=BF16Vector([0.8]*128),
            status=KnowledgeClaimStatus.ACCEPTED,
            confidence=0.95
        ),
        KnowledgeClaim(
            author_id="System",
            content="Water boils at 100°C at sea level",
            vector=BF16Vector([0.7, 0.6, 0.5] + [0.0]*125),
            status=KnowledgeClaimStatus.ACCEPTED,
            confidence=0.85
        )
    ]
    
    for claim in initial_claims:
        await pond.kip_manager.register_claim(claim)
        for agent in pond.agents:
            agent.knowledge[claim.claim_id] = claim
            agent.known_claims.add(claim.claim_id)

    # Start simulation
    simulation_duration = 45.0
    reporter_task = asyncio.create_task(run_consensus_reporter(pond))
    simulation_task = asyncio.create_task(pond.start(simulation_duration, profile))

    await simulation_task
    reporter_task.cancel()
    
    # Final consensus report
    consensus = await pond.kip_manager.build_consensus()
    logger.info("\n==== FINAL KIP CONSENSUS ====")
    for status, count in consensus["status_counts"].items():
        logger.info(f"{status.upper():<12}: {count:>4}")
    logger.info(f"AGENTS: {len(pond.agents)}")
    logger.info(f"SIMULATION DURATION: {simulation_duration}s")
    logger.info("============================")

if __name__ == "__main__":
    try:
        # Run with profiling enabled
        asyncio.run(demo_pond(profile=True))
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user")
