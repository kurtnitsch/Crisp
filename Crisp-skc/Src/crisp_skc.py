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
from functools import cached_property

# Configure logging with better performance
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import ICP agent components
from ic_agent import Agent, Principal, HttpAgent, Identity
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
        if self._data_hash is None and (self.payload or self.binary_payload):
            self._data_hash = self._compute_hash_fast()
        return self._data_hash

    @data_hash.setter
    def data_hash(self, value: str):
        self._data_hash = value

    def _compute_hash_fast(self) -> str:
        """Optimized hash computation"""
        if self.binary_payload:
            return hashlib.blake2b(self.binary_payload, digest_size=8).hexdigest()
        elif self.payload:
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
        if self.binary_data:
            return hashlib.blake2b(self.binary_data, digest_size=16).hexdigest()
        elif self.data:
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
                raise ValueError(f"Invalid resource
