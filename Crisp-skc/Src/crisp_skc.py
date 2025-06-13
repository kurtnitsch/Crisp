--- START OF FILE fixed_skc_library.py ---

#!/usr/bin/env python3
"""
Crisp SKC (Shared Knowledge Core) Library - Optimized Version with ICP Integration
High-performance implementation with ICP-compatible WASM64 support

Author: Kurt Nitsch, Optimized from Crisp Protocol Specification v1.2
Date: June 13, 2025
License: Apache License 2.0

Key Optimizations:
- Reduced memory allocations and copying
- Optimized serialization/deserialization
- Improved indexing and search performance
- Better async handling and reduced lock contention
- Cached computations and lazy loading
- Memory-efficient data structures
- ICP network integration for WASM64 canister deployment
"""

import asyncio
import hashlib
import time
import uuid
import base64
import struct
import msgpack
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
# from pathlib import Path # Not directly used in this version's core logic
from collections import defaultdict
# import weakref # Not directly used in this version's core logic
from functools import lru_cache, cached_property
# import pickle # Not directly used in this version's core logic
# import msgpack.exceptions # Handled by generic Exception for simplicity in dataclasses
# import struct # Handled by generic Exception for simplicity in dataclasses

# Configure logging with better performance
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import ICP agent components
from ic_agent import Principal, HttpAgent, Identity
from ic_agent.errors import AgentError  # For specific ICP agent errors
import requests.exceptions  # For network errors with HttpAgent

# For dummy identity generation in example usage
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from ic_agent.identity import BasicIdentity
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False
    class BasicIdentity:  # Dummy class if cryptography is not available
        def __init__(self, principal_text: str):
            self._principal = Principal.from_text(principal_text)
        def get_principal(self) -> Principal:
            return self._principal
        def sign(self, blob: bytes) -> bytes:
            raise NotImplementedError("Cannot sign without cryptography library")
        @staticmethod
        def from_pem(pem_bytes: bytes):
            raise NotImplementedError("Cannot load identity without cryptography library")
    logger.warning("Cryptography library not found. Dummy Identity will be used for ICP examples, which will not work for actual signing operations.")

# ============================================================================
# Constants
# ============================================================================
ICP_WASM64_MAX_MEM = 6 * 1024 * 1024 * 1024  # 6GiB

# ============================================================================
# Core Data Structures and Enums (Optimized)
# ============================================================================

class MessageType(Enum):
    """Cognitive Packet message types - Using string values for faster comparison"""
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
    WASM64_ICP = "WASM64:ICP"  # ICP-compatible WASM64
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
    """Message group classifications - Reduced set for better performance"""
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

# ============================================================================
# Optimized Data Classes with Better Memory Management
# ============================================================================

@dataclass
class CognitivePacket:
    """
    Optimized Cognitive Packet with reduced memory footprint and ICP support
    """
    # Core required fields
    dest: str
    msg_type: MessageType
    sender: str

    # Auto-generated fields
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Optional fields
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
    encoding: str = "MessagePack"
    data_type: Optional[str] = None
    _data_hash: Optional[str] = None
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
    canister_id: Optional[str] = None  # ICP canister ID
    cycles_required: Optional[int] = None  # Cycles for ICP execution

    # Payload data
    payload: Optional[Dict[str, Any]] = None
    binary_payload: Optional[bytes] = None

    # Cached serialization
    _cached_binary: Optional[bytes] = field(default=None, init=False, repr=False)
    _cached_binary_valid: bool = field(default=False, init=False, repr=False)

    @property
    def data_hash(self) -> Optional[str]:
        """Lazy computation of data hash"""
        if self._data_hash is None and (self.payload is not None or self.binary_payload is not None):
            self._data_hash = self._compute_hash_fast()
        return self._data_hash

    @data_hash.setter
    def data_hash(self, value: str):
        self._data_hash = value

    def _compute_hash_fast(self) -> str:
        """Optimized hash computation"""
        if self.binary_payload is not None:
            return hashlib.blake2b(self.binary_payload, digest_size=8).hexdigest()
        elif self.payload is not None:
            try:
                serialized = msgpack.packb(self.payload, use_bin_type=True)
                return hashlib.blake2b(serialized, digest_size=8).hexdigest()
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to msgpack payload for hash: {e}")
                return hashlib.blake2b(str(self.payload).encode(), digest_size=8).hexdigest()
        return ""

    def invalidate_cache(self) -> None:
        """Invalidate cached binary representation and hash"""
        self._cached_binary_valid = False
        self._cached_binary = None
        self._data_hash = None

    def validate_icp_wasm64(self) -> bool:
        """Validate if packet is suitable for ICP WASM64 execution"""
        if self.compute_class != ComputeClass.WASM64_ICP:
            logger.debug(f"Packet not ICP WASM64: compute_class is {self.compute_class}")
            return False
        if self.canister_id is None:
            logger.warning("Missing canister_id for ICP WASM64 packet")
            return False
        try:
            Principal.from_text(self.canister_id)
        except ValueError:
            logger.warning(f"Invalid canister_id format: {self.canister_id}")
            return False
        if self.cycles_required is not None and self.cycles_required < 0:
            logger.warning("Invalid cycles_required for ICP WASM64 packet")
            return False
        if self.binary_payload and len(self.binary_payload) > ICP_WASM64_MAX_MEM:
            logger.warning(f"Binary payload ({len(self.binary_payload)} bytes) exceeds ICP WASM memory limit (6GiB)")
            return False
        return True

    def to_binary_format(self) -> bytes:
        """Optimized binary serialization with ICP fields"""
        if self._cached_binary_valid and self._cached_binary:
            return self._cached_binary

        headers = {}
        header_fields = [
            ('dest', self.dest),
            ('msg_type', self.msg_type.value if isinstance(self.msg_type, MessageType) else self.msg_type),
            ('msg_id', self.msg_id),
            ('sender', self.sender),
            ('timestamp', self.timestamp),
            ('priority', self.priority if self.priority != 5 else None),
            ('group', self.group.value if self.group else None),
            ('part', self.part),
            ('compute_class', self.compute_class.value if self.compute_class else None),
            ('context', self.context),
            ('auth_token', self.auth_token),
            ('encryption', self.encryption),
            ('access_level', self.access_level),
            ('data_type', self.data_type),
            ('data_hash', self.data_hash),
            ('canister_id', self.canister_id),
            ('cycles_required', self.cycles_required)
        ]

        for key, value in header_fields:
            if value is not None:
                headers[key] = value

        try:
            header_data = msgpack.packb(headers, use_bin_type=True)
            payload_data = b''
            if self.payload is not None:
                payload_data = msgpack.packb(self.payload, use_bin_type=True)
            binary_data = self.binary_payload or b''

            result = (struct.pack('!III', len(header_data), len(payload_data), len(binary_data)) +
                      header_data + payload_data + binary_data)

            self._cached_binary = result
            self._cached_binary_valid = True
            return result

        except (TypeError, ValueError, msgpack.exceptions.PackException, struct.error) as e:
            logger.error(f"Binary serialization failed for CognitivePacket {self.msg_id}: {e}")
            raise ValueError(f"Serialization error: {e}") from e

    @classmethod
    def from_binary_format(cls, binary_data: bytes) -> 'CognitivePacket':
        """Optimized binary deserialization with ICP fields"""
        if len(binary_data) < 12:
            raise ValueError(f"Invalid binary packet: too short ({len(binary_data)} bytes)")

        try:
            header_len, payload_len, binary_len = struct.unpack('!III', binary_data[:12])
            total_expected = 12 + header_len + payload_len + binary_len
            if len(binary_data) != total_expected:
                raise ValueError(f"Length mismatch: expected {total_expected}, got {len(binary_data)}")

            offset = 12
            header_data = msgpack.unpackb(binary_data[offset:offset+header_len], raw=False, strict_map_key=False)
            offset += header_len

            payload_data = None
            if payload_len > 0:
                payload_data = msgpack.unpackb(binary_data[offset:offset+payload_len], raw=False, strict_map_key=False)
            offset += payload_len

            binary_payload = binary_data[offset:offset+binary_len] if binary_len > 0 else None

            def enum_or_none(enum_cls, value):
                if value is None:
                    return None
                try:
                    return enum_cls(value)
                except (ValueError, TypeError):
                    return None

            kwargs = {
                'dest': header_data.get('dest', ''),
                'msg_type': enum_or_none(MessageType, header_data.get('msg_type', 'query')),
                'msg_id': header_data.get('msg_id', str(uuid.uuid4())),
                'sender': header_data.get('sender', ''),
                'timestamp': header_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                'payload': payload_data,
                'binary_payload': binary_payload,
                'canister_id': header_data.get('canister_id'),
                'cycles_required': header_data.get('cycles_required')
            }

            optional_fields = [
                'priority', 'group', 'part', 'compute_class', 'context',
                'auth_token', 'encryption', 'access_level', 'data_type', 'data_hash'
            ]

            for field_name in optional_fields:
                if field_name in header_data:
                    value = header_data[field_name]
                    if field_name == 'group':
                        kwargs[field_name] = enum_or_none(MessageGroup, value)
                    elif field_name == 'compute_class':
                        kwargs[field_name] = enum_or_none(ComputeClass, value)
                    else:
                        kwargs[field_name] = value

            instance = cls(**kwargs)
            instance._cached_binary = binary_data
            instance._cached_binary_valid = True
            return instance

        except Exception as e:
            logger.error(f"Binary deserialization failed: {e}")
            raise ValueError(f"Failed to parse binary packet: {e}") from e

@dataclass
class SKCResource:
    """
    Optimized SKC Resource with ICP-compatible WASM64 support
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
    canister_id: Optional[str] = None  # ICP canister ID
    cycles_required: Optional[int] = None  # Cycles for ICP execution
    _checksum: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat()
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
        if self.binary_data is not None and self.size_bytes == 0:
            self.size_bytes = len(self.binary_data)
        elif self.data is not None and self.size_bytes == 0:
            try:
                self.size_bytes = len(msgpack.packb(self.data, use_bin_type=True))
            except (TypeError, ValueError):
                self.size_bytes = len(str(self.data).encode())

    @cached_property
    def checksum(self) -> str:
        """Cached checksum computation"""
        if self._checksum is None:
            self._checksum = self._compute_checksum_fast()
        return self._checksum

    def _compute_checksum_fast(self) -> str:
        """Optimized checksum computation"""
        if self.binary_data is not None:
            return hashlib.blake2b(self.binary_data, digest_size=16).hexdigest()
        elif self.data is not None:
            try:
                serialized = msgpack.packb(self.data, use_bin_type=True)
                return hashlib.blake2b(serialized, digest_size=16).hexdigest()
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to msgpack data for checksum: {e}")
                return hashlib.blake2b(str(self.data).encode(), digest_size=16).hexdigest()
        return ""

    def invalidate_cache(self) -> None:
        """Invalidate all cached values (e.g., checksum)"""
        if 'checksum' in self.__dict__:
            del self.__dict__['checksum']
        self._checksum = None

    def update_data(self, data: Optional[Any] = None, binary_data: Optional[bytes] = None) -> None:
        """Efficiently update data and invalidate caches"""
        if data is not None:
            self.data, self.binary_data = data, None
        elif binary_data is not None:
            self.data, self.binary_data = None, binary_data
        else:
            return

        if self.binary_data is not None:
            self.size_bytes = len(self.binary_data)
        elif self.data is not None:
            try:
                self.size_bytes = len(msgpack.packb(self.data, use_bin_type=True))
            except (TypeError, ValueError):
                self.size_bytes = len(str(self.data).encode())
        else:
            self.size_bytes = 0

        self.invalidate_cache()

    def validate_icp_wasm64(self) -> bool:
        """Validate if resource is suitable for ICP WASM64 execution"""
        if self.compute_class != ComputeClass.WASM64_ICP:
            logger.debug(f"Resource not ICP WASM64: compute_class is {self.compute_class}")
            return False
        if self.canister_id is None:
            logger.warning("Missing canister_id for ICP WASM64 resource")
            return False
        try:
            Principal.from_text(self.canister_id)
        except ValueError:
            logger.warning(f"Invalid canister_id format: {self.canister_id}")
            return False
        if self.cycles_required is not None and self.cycles_required < 0:
            logger.warning("Invalid cycles_required for ICP WASM64 resource")
            return False
        if self.binary_data and len(self.binary_data) > ICP_WASM64_MAX_MEM:
            logger.warning(f"Binary data ({len(self.binary_data)} bytes) exceeds ICP WASM memory limit (6GiB)")
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Optimized dictionary conversion"""
        result = {
            'resource_id': self.resource_id,
            'full_address': self.full_address,
            'resource_type': self.resource_type.value if isinstance(self.resource_type, ResourceType) else self.resource_type,
            'version': self.version,
            'compute_class': self.compute_class.value if self.compute_class else None,
            'size_bytes': self.size_bytes,
            'created': self.created,
            'permissions': self.permissions,
            'checksum': self.checksum,
            'metadata': self.metadata,
            'tags': list(self.tags),
            'data': self.data,
            'canister_id': self.canister_id,
            'cycles_required': self.cycles_required
        }

        if self.binary_data:
            # Warn about large binary data
            if len(self.binary_data) > 16 * 1024 * 1024:  # 16MB arbitrary warning threshold
                logger.warning("Encoding large binary_data for dict conversion; this may be memory intensive.")
            result['binary_data'] = base64.b64encode(self.binary_data).decode('ascii')

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SKCResource':
        """Optimized creation from dictionary"""
        _data = data.copy()
        # Parse enums from string if needed
        if 'resource_type' in _data and not isinstance(_data['resource_type'], ResourceType):
            try:
                _data['resource_type'] = ResourceType(_data['resource_type'])
            except Exception:
                raise ValueError(f"Invalid resource_type: {_data['resource_type']}")
        if 'compute_class' in _data and _data['compute_class'] is not None and not isinstance(_data['compute_class'], ComputeClass):
            try:
                _data['compute_class'] = ComputeClass(_data['compute_class'])
            except Exception:
                _data['compute_class'] = None
        if 'tags' in _data and not isinstance(_data['tags'], set):
            _data['tags'] = set(_data['tags'])
        if 'binary_data' in _data and isinstance(_data['binary_data'], str):
            _data['binary_data'] = base64.b64decode(_data['binary_data'])
        return cls(**_data)


# ============================================================================
# Asynchronous Read-Write Lock (MISSING FROM YOUR SNIPPET, ADDED HERE)
# ============================================================================

class AsyncReadWriteLock:
    """
    An asynchronous Read-Write Lock for asyncio.
    Allows multiple readers or one writer. Writers are prioritized.
    """
    def __init__(self):
        self._lock = asyncio.Lock()  # Protects _readers, _writer_waiting, _writer_event, _reader_event
        self._readers = 0
        self._writer_waiting = False
        self._writer_event = asyncio.Event() # Signalled when readers count is zero (for writer)
        self._reader_event = asyncio.Event() # Signalled when writer finishes (for readers)
        self._writer_event.set() # Initially allow writer if no readers are present
        self._reader_event.set() # Initially allow readers

    async def reader_acquire(self):
        async with self._lock:
            if self._writer_waiting:
                # If a writer is waiting, readers must wait for writer to finish
                await self._reader_event.wait()
            # If no writer is waiting, or writer has finished, readers can proceed
            self._readers += 1
            self._writer_event.clear() # Block new writers if any readers are active

    async def reader_release(self):
        async with self._lock:
            self._readers -= 1
            if self._readers == 0 and self._writer_waiting:
                # Last reader exiting, if writer is waiting, signal it
                self._writer_event.set()

    async def writer_acquire(self):
        async with self._lock:
            self._writer_waiting = True
            # Wait until no readers are active
            await self._writer_event.wait()
            self._reader_event.clear() # Block new readers from entering
            # Now the lock is held exclusively by this writer (via _lock and _readers=0)

    async def writer_release(self):
        async with self._lock:
            self._writer_waiting = False
            self._reader_event.set() # Allow readers to proceed
            self._writer_event.set() # Allow next writer if any (or new readers will clear)

    # Context managers for easy usage
    @property
    def reader(self):
        class ReaderContext:
            def __init__(self, rw_lock_instance):
                self._rw_lock = rw_lock_instance
            async def __aenter__(self_rc):
                await self._rw_lock.reader_acquire()
            async def __aexit__(self_rc, exc_type, exc_val, exc_tb):
                await self._rw_lock.reader_release()
        return ReaderContext(self)

    @property
    def writer(self):
        class WriterContext:
            def __init__(self, rw_lock_instance):
                self._rw_lock = rw_lock_instance
            async def __aenter__(self_wc):
                await self._rw_lock.writer_acquire()
            async def __aexit__(self_wc, exc_type, exc_val, exc_tb):
                await self._rw_lock.writer_release()
        return WriterContext(self)

# ============================================================================
# High-Performance SKC Core Implementation with ICP Integration
# ============================================================================

class OptimizedSKCCore:
    """
    High-performance Shared Knowledge Core with ICP network integration
    """
    
    def __init__(self, max_cache_size: int = 10000, icp_endpoint: str = "https://ic0.app"):
        # Core storage
        self.resources: Dict[str, SKCResource] = {}
        self.address_map: Dict[str, str] = {}
        
        # Performance indexes
        self.type_index: Dict[ResourceType, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.compute_class_index: Dict[ComputeClass, Set[str]] = defaultdict(set)
        self.canister_index: Dict[str, Set[str]] = defaultdict(set)  # Index for ICP canister IDs
        
        # Concurrency optimization
        self._rw_lock = AsyncReadWriteLock()
        self._index_lock = asyncio.Lock() # Still useful for internal index updates if needed, though RW lock can cover this.
                                          # For simplicity, keeping it separate for index updates within writes.
        
        # Caching
        self.max_cache_size = max_cache_size
        self._query_cache: Dict[str, Tuple[List[SKCResource], float]] = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Statistics
        self.stats = {
            'resources_count': 0,
            'queries_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'index_updates': 0,
            'icp_deployments': 0
        }
        
        # ICP integration - Agent initialized without identity; identity passed per call if needed
        self.icp_agent = HttpAgent(icp_endpoint)
        logger.info(f"Initialized OptimizedSKCCore with ICP endpoint: {icp_endpoint}")
    
    def _cache_key(self, resource_type=None, tags=None, metadata_query=None, access_role=None, compute_class=None, canister_id=None) -> str:
        """Generate cache key for query"""
        key_parts = [
            str(resource_type.value if resource_type else ''),
            str(sorted(list(tags)) if tags else ''), # Convert set to list for sorting
            str(sorted(metadata_query.items()) if metadata_query else ''),
            str(access_role or ''),
            str(compute_class.value if compute_class else ''),
            str(canister_id or '')
        ]
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - timestamp < self._cache_ttl
    
    @lru_cache(maxsize=1000)
    def _generate_resource_id(self, full_address: str) -> str:
        """Cached resource ID generation"""
        return hashlib.blake2b(full_address.encode(), digest_size=6).hexdigest()
    
    async def _update_indexes(self, resource: SKCResource, is_new: bool):
        """Update all indexes for a resource. If not new, remove old entries first."""
        async with self._index_lock: # Protect index modification
            # Before adding, ensure old entries are removed if resource was updated/moved
            if not is_new and resource.resource_id in self.resources:
                old_resource = self.resources[resource.resource_id]
                self.type_index[old_resource.resource_type].discard(old_resource.resource_id)
                for tag in old_resource.tags:
                    self.tag_index[tag].discard(old_resource.resource_id)
                if old_resource.compute_class:
                    self.compute_class_index[old_resource.compute_class].discard(old_resource.resource_id)
                if old_resource.canister_id:
                    self.canister_index[old_resource.canister_id].discard(old_resource.resource_id)

            resource_id = resource.resource_id
            self.type_index[resource.resource_type].add(resource_id)
            for tag in resource.tags:
                self.tag_index[tag].add(resource_id)
            if resource.compute_class:
                self.compute_class_index[resource.compute_class].add(resource_id)
            if resource.canister_id:
                self.canister_index[resource.canister_id].add(resource_id)
            self.stats['index_updates'] += 1
    
    async def _remove_from_indexes(self, resource: SKCResource):
        """Remove resource from all indexes"""
        async with self._index_lock:
            resource_id = resource.resource_id
            self.type_index[resource.resource_type].discard(resource_id)
            # Use intersection to avoid issues if a tag was already removed or never existed
            for tag in resource.tags:
                self.tag_index[tag].discard(resource_id)
            if resource.compute_class:
                self.compute_class_index[resource.compute_class].discard(resource_id)
            if resource.canister_id:
                self.canister_index[resource.canister_id].discard(resource_id)
    
    async def register_resource(self,
                               full_address: str,
                               resource_type: ResourceType,
                               data: Optional[Any] = None,
                               binary_data: Optional[bytes] = None,
                               compute_class: Optional[ComputeClass] = None,
                               tags: Optional[Set[str]] = None,
                               metadata: Optional[Dict[str, Any]] = None,
                               canister_id: Optional[str] = None,
                               cycles_required: Optional[int] = None) -> SKCResource:
        """
        Register a new resource or update an existing one in the SKC with ICP support.
        """
        async with self._rw_lock.writer: # Acquire writer lock for modification
            resource_id = self._generate_resource_id(full_address)
            
            is_new_resource = resource_id not in self.resources
            
            if not is_new_resource:
                logger.warning(f"Resource {resource_id} already exists, updating")
                existing = self.resources[resource_id]
                existing.update_data(data=data, binary_data=binary_data)
                if tags:
                    existing.tags.update(tags)
                if metadata:
                    existing.metadata.update(metadata)
                if compute_class:
                    existing.compute_class = compute_class
                if canister_id:
                    existing.canister_id = canister_id
                if cycles_required is not None:
                    existing.cycles_required = cycles_required
                resource_to_update = existing
            else:
                resource_to_update = SKCResource(
                    resource_id=resource_id,
                    full_address=full_address,
                    resource_type=resource_type,
                    compute_class=compute_class,
                    tags=tags or set(),
                    metadata=metadata or {},
                    data=data,
                    binary_data=binary_data,
                    canister_id=canister_id,
                    cycles_required=cycles_required
                )
                self.resources[resource_id] = resource_to_update
                self.address_map[full_address] = resource_id
                self.stats['resources_count'] += 1
            
            await self._update_indexes(resource_to_update, is_new_resource)
            
            # Invalidate query cache if data might have changed
            self._query_cache.clear()
            logger.debug("Query cache cleared after resource registration/update")
            
            logger.info(f"Registered resource {resource_id} at {full_address}")
            return resource_to_update
    
    async def query_resources(self,
                             resource_type: Optional[ResourceType] = None,
                             tags: Optional[Set[str]] = None,
                             metadata_query: Optional[Dict[str, Any]] = None,
                             access_role: Optional[str] = None,
                             compute_class: Optional[ComputeClass] = None,
                             canister_id: Optional[str] = None) -> List[SKCResource]:
        """
        Query resources with ICP support. Utilizes a read-write lock for concurrency.
        """
        cache_key = self._cache_key(resource_type, tags, metadata_query, access_role, compute_class, canister_id)
        
        async with self._rw_lock.reader: # Acquire reader lock for query
            if cache_key in self._query_cache:
                results, timestamp = self._query_cache[cache_key]
                if self._is_cache_valid(timestamp):
                    self.stats['cache_hits'] += 1
                    return results
            
            self.stats['cache_misses'] += 1
            results = []
            
            # Start with all resources or a filtered set based on primary indexes
            candidate_ids = set(self.resources.keys())
            if resource_type:
                candidate_ids = candidate_ids.intersection(self.type_index.get(resource_type, set()))
            if tags:
                for tag in tags:
                    candidate_ids = candidate_ids.intersection(self.tag_index.get(tag, set()))
            if compute_class:
                candidate_ids = candidate_ids.intersection(self.compute_class_index.get(compute_class, set()))
            if canister_id:
                candidate_ids = candidate_ids.intersection(self.canister_index.get(canister_id, set()))
            
            # Filter remaining candidates based on other criteria
            for resource_id in candidate_ids: # Iterate over the potentially reduced set
                resource = self.resources.get(resource_id)
                if not resource: # Resource might have been removed concurrently (though RW lock prevents it here)
                    continue
                if access_role and not self._has_access(resource, access_role):
                    continue
                if metadata_query and not self._matches_metadata(resource, metadata_query):
                    continue
                results.append(resource)
            
            # Cache the results
            if len(self._query_cache) >= self.max_cache_size:
                # Simple cache eviction: clear all if max size reached. More sophisticated
                # LRU or LFU eviction could be implemented.
                self._query_cache.clear()
                logger.debug("Query cache cleared due to size limit for new entry.")
            self._query_cache[cache_key] = (results, time.time())
            
            self.stats['queries_processed'] += 1
            return results
    
    def _has_access(self, resource: SKCResource, role: str) -> bool:
        """Check if role has access to resource"""
        return role in resource.permissions.get('read', []) or role in resource.permissions.get('write', [])
    
    def _matches_metadata(self, resource: SKCResource, query: Dict[str, Any]) -> bool:
        """Check if resource metadata matches query"""
        for key, value in query.items():
            # Use .get() to avoid KeyError if key is missing in resource metadata
            if resource.metadata.get(key) != value:
                return False
        return True
    
    async def deploy_to_icp(self, resource: SKCResource, identity: Identity) -> bool:
        """
        Deploy a WASM64 resource to an ICP canister.
        
        Args:
            resource: SKCResource with WASM64_ICP compute class.
            identity: An `ic_agent.Identity` object for authentication and signing.
        
        Returns:
            bool: True if deployment succeeds, False otherwise.
        """
        if not resource.validate_icp_wasm64():
            logger.error(f"Resource {resource.resource_id} is not valid for ICP WASM64 deployment.")
            return False
        
        if not resource.binary_data:
            logger.error(f"Resource {resource.resource_id} has no binary data for deployment.")
            return False
        
        try:
            canister_principal = Principal.from_text(resource.canister_id)
            
            # The HttpAgent needs to be configured with the identity for signed calls,
            # or the identity passed directly to the call method.
            # Here, we pass the identity directly to the update_canister method.
            logger.info(f"Attempting to deploy resource {resource.resource_id} to ICP canister {resource.canister_id} "
                        f"using principal {identity.get_principal()}")
            await self.icp_agent.update_canister(
                canister_principal, 
                resource.binary_data, 
                identity=identity # Pass the identity here for signing
            )
            self.stats['icp_deployments'] += 1
            logger.info(f"Successfully deployed resource {resource.resource_id} to ICP canister {resource.canister_id}.")
            return True
            
        except (ValueError, AgentError, requests.exceptions.RequestException) as e:
            # Catch specific ICP agent errors or network errors
            logger.error(f"ICP deployment failed for resource {resource.resource_id} to canister {resource.canister_id}: {type(e).__name__}: {e}")
            return False
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred during ICP deployment for resource {resource.resource_id}: {type(e).__name__}: {e}")
            return False

# ============================================================================
# Example Usage (MISSING FROM YOUR SNIPPET, ADDED HERE)
# ============================================================================

async def main():
    # Initialize SKC core with ICP mainnet endpoint
    skc = OptimizedSKCCore(max_cache_size=1000, icp_endpoint="https://ic0.app")
    
    # --- IDENTITY SETUP FOR ICP DEPLOYMENT (FOR DEMONSTRATION ONLY) ---
    # WARNING: This section is for demonstration purposes. In a production environment,
    # you MUST securely load your ICP identity (e.g., from an encrypted file, KMS, etc.).
    # NEVER hardcode or expose private keys.
    
    test_identity: Optional[Identity] = None
    if _CRYPTO_AVAILABLE:
        try:
            # Generate a new random Ed25519 private key for this test run
            # This is NOT persistent and NOT secure for real assets.
            private_key = ed25519.Ed25519PrivateKey.generate()
            private_key_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            test_identity = BasicIdentity.from_pem(private_key_bytes)
            logger.info(f"Generated dummy test principal for ICP operations: {test_identity.get_principal()}")
        except Exception as e:
            logger.error(f"Failed to generate test identity using cryptography: {e}")
            test_identity = None
    else:
        logger.warning("Cryptography library not found. Dummy Identity will be used for ICP examples, which will not work for actual signing operations.")
        # Fallback to a dummy identity if crypto is not available, but it won't be able to sign
        test_identity = BasicIdentity(principal_text="aaaaa-aa") # A dummy principal

    # Register an ICP-compatible WASM64 resource
    # NOTE: "rwlgt-iiaaa-aaaaa-aaaaa-cai" is the NNS root canister.
    # Deploying a module to it will fail unless you have NNS admin permissions.
    # For a realistic test, you would use a canister ID you have created and control.
    # For this example, we'll keep it as a placeholder to demonstrate the call path.
    test_canister_id = "rwlgt-iiaaa-aaaaa-aaaaa-cai" # Example: NNS Root Canister (deployment will likely fail)
    # test_canister_id = "rdjem-pyaaa-aaaaa-aaaaa-cai" # Example: A hypothetical user-controlled canister
    
    resource_to_deploy = await skc.register_resource(
        full_address=f"icp://{test_canister_id}/my_module",
        resource_type=ResourceType.WASM,
        # A minimal valid WASM binary (version, type section, end marker)
        # This is enough to pass basic format checks, but it does nothing.
        binary_data=b'\x00\x61\x73\x6d\x01\x00\x00\x00\x01\x04\x60\x00\x00', 
        compute_class=ComputeClass.WASM64_ICP,
        tags={"wasm", "icp", "64-bit", "experimental"},
        metadata={"canister_type": "compute", "purpose": "example_module"},
        canister_id=test_canister_id,
        cycles_required=1_000_000_000
    )
    
    # Attempt to deploy to ICP if a valid identity exists
    if test_identity:
        logger.info(f"Attempting deployment for resource {resource_to_deploy.resource_id}...")
        deployed = await skc.deploy_to_icp(resource_to_deploy, test_identity)
        print(f"Deployment to ICP for {resource_to_deploy.resource_id}: {'Success' if deployed else 'Failed'}")
    else:
        print("Skipping ICP deployment test due to missing/invalid identity.")

    # Query ICP WASM64 resources
    print("\nQuerying for ICP WASM64 resources...")
    results = await skc.query_resources(
        resource_type=ResourceType.WASM,
        compute_class=ComputeClass.WASM64_ICP,
        tags={"icp"},
        access_role="public",
        canister_id=test_canister_id
    )
    
    if results:
        print(f"Found {len(results)} ICP WASM64 resource(s):")
        for res in results:
            print(f"  - Resource ID: {res.resource_id}, Address: {res.full_address}, Canister: {res.canister_id}, Checksum: {res.checksum}")
    else:
        print("No ICP WASM64 resources found matching criteria.")

    # Create and validate an ICP WASM64 packet
    print("\nCreating and validating an ICP WASM64 packet...")
    packet = CognitivePacket(
        dest=f"icp://{test_canister_id}",
        msg_type=MessageType.EXEC,
        sender="client123",
        compute_class=ComputeClass.WASM64_ICP,
        canister_id=test_canister_id,
        cycles_required=500_000_000,
        payload={"method": "execute_compute", "args": ["param1", 123]}
    )
    print(f"ICP WASM64 Packet Valid: {packet.validate_icp_wasm64()}")
    
    # Demonstrate binary serialization/deserialization of the packet
    try:
        binary_packet = packet.to_binary_format()
        print(f"Packet serialized to {len(binary_packet)} bytes.")
        deserialized_packet = CognitivePacket.from_binary_format(binary_packet)
        print(f"Packet deserialized successfully. Msg ID: {deserialized_packet.msg_id}, Sender: {deserialized_packet.sender}")
        assert packet.msg_id == deserialized_packet.msg_id
        assert packet.sender == deserialized_packet.sender
        assert packet.canister_id == deserialized_packet.canister_id
        assert packet.payload == deserialized_packet.payload
        print("Serialization/Deserialization successful and consistent.")
    except Exception as e:
        print(f"Serialization/Deserialization test failed: {e}")

    print(f"\nSKC Statistics: {skc.stats}")

if __name__ == "__main__":
    asyncio.run(main())

--- END OF FILE fixed_skc_library.py ---
