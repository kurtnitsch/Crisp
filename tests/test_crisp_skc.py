#!/usr/bin/env python3
"""
Unit and integration tests for Crisp SKC Library
Tests CognitivePacket, SKCResource, and OptimizedSKCCore functionality
"""

import unittest
import asyncio
import msgpack
from datetime import datetime, timezone
from src.crisp_skc import (
    OptimizedSKCCore,
    CognitivePacket,
    SKCResource,
    MessageType,
    ResourceType,
    ComputeClass,
    MessageGroup
)

class TestCrispSKC(unittest.TestCase):
    def setUp(self):
        """Initialize SKC and event loop for async tests"""
        self.skc = OptimizedSKCCore(max_cache_size=100)
        self.loop = asyncio.get_event_loop()
    
    def tearDown(self):
        """Clean up resources"""
        self.skc.resources.clear()
        self.skc.address_map.clear()
        self.skc.type_index.clear()
        self.skc.tag_index.clear()
        self.skc.compute_class_index.clear()
        self.skc._query_cache.clear()

    def run_async(self, coro):
        """Helper to run async tests"""
        return self.loop.run_until_complete(coro)

    # Unit Tests for CognitivePacket
    def test_cognitive_packet_serialization(self):
        """Test packet serialization and deserialization"""
        cp = CognitivePacket(
            dest="X12.09.78",
            msg_type=MessageType.QUERY,
            sender="E13.54.78",
            payload={"tags": ["test"]},
            data_type="data"
        )
        binary = cp.to_binary_format()
        cp2 = CognitivePacket.from_binary_format(binary)
        self.assertEqual(cp.dest, cp2.dest)
        self.assertEqual(cp.msg_type, cp2.msg_type)
        self.assertEqual(cp.payload, cp2.payload)
        self.assertEqual(cp.data_hash, cp2.data_hash)

    def test_cognitive_packet_invalid_binary(self):
        """Test handling of invalid binary packet"""
        with self.assertRaises(ValueError):
            CognitivePacket.from_binary_format(b"invalid")
    
    def test_cognitive_packet_data_hash(self):
        """Test data hash computation"""
        cp = CognitivePacket(
            dest="X12", msg_type=MessageType.DATA, sender="E13",
            payload={"data": "test"}
        )
        hash1 = cp.data_hash
        self.assertIsNotNone(hash1)
        cp.payload = {"data": "test2"}
        cp.invalidate_cache()
        hash2 = cp.data_hash
        self.assertNotEqual(hash1, hash2)

    # Unit Tests for SKCResource
    def test_skc_resource_checksum(self):
        """Test resource checksum computation"""
        resource = SKCResource(
            resource_id="123",
            full_address="https://example.com/test",
            resource_type=ResourceType.DATA,
            data={"value": 42}
        )
        checksum1 = resource.checksum
        resource.update_data(data={"value": 43})
        checksum2 = resource.checksum
        self.assertNotEqual(checksum1, checksum2)

    def test_skc_resource_to_dict(self):
        """Test resource to_dict conversion"""
        resource = SKCResource(
            resource_id="123",
            full_address="https://example.com/test",
            resource_type=ResourceType.DATA,
            binary_data=b"test"
        )
        data = resource.to_dict()
        self.assertEqual(data["binary_data"], "dGVzdA==")  # base64 of "test"
        self.assert
