class SerializationError(Exception):
    """Custom exception for serialization failures"""
    CODES = {
        100: "Header serialization failure",
        101: "Payload serialization failure",
        200: "Header deserialization failure",
        201: "Payload deserialization failure",
        300: "Invalid binary structure"
    }
    
    def __init__(self, message, code=0):
        super().__init__(message)
        self.code = code
        self.message = f"{message} (Error #{code}: {self.CODES.get(code, 'Unknown error')})"

class CryptoError(Exception):
    """Custom exception for cryptographic operations"""
    CODES = {
        100: "Invalid key format",
        101: "Invalid signature format",
        200: "Signing operation failed",
        201: "Verification operation failed"
    }
    
    def __init__(self, message, code=0):
        super().__init__(message)
        self.code = code
        self.message = f"{message} (Error #{code}: {self.CODES.get(code, 'Unknown error')})"

# In serialization methods:
def to_binary_format(self) -> bytes:
    try:
        # ... serialization code ...
    except msgpack.PackException as e:
        raise SerializationError(f"Msgpack packing failed: {str(e)}", 100) from e
    except struct.error as e:
        raise SerializationError(f"Binary packing failed: {str(e)}", 300) from e
    except Exception as e:
        raise SerializationError(f"Unexpected serialization error: {str(e)}", 999) from e
