class VetKey:
    """
    Implementation of forward-secure cryptographic keys using the VetKey protocol.
    
    VetKey provides:
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
    # ... implementation ...

class CognitivePacket:
    """
    Fundamental network packet structure for the Crisp protocol.
    
    The packet uses a binary format with three sections:
    1. Header section (msgpack serialized metadata)
    2. Payload section (msgpack serialized structured data)
    3. Binary payload section (raw binary data)
    
    The packet supports:
    - Cryptographic signatures using VetKey
    - Automatic caching of serialized form
    - Validation of hop count and signatures
    
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
    # ... implementation ...
