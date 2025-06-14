class VetKey:
    # ... existing implementation ...
    
    def _validate_signature_params(self, signature, data, public_key_bytes):
        """Validate signature parameters before verification"""
        if not isinstance(signature, bytes) or len(signature) != 64:
            raise CryptoError("Invalid signature format")
        if not isinstance(data, bytes):
            raise CryptoError("Data must be bytes")
        if not isinstance(public_key_bytes, bytes) or len(public_key_bytes) != 32:
            raise CryptoError("Invalid public key format")

    @staticmethod
    def verify(signature: bytes, data: bytes, public_key_bytes: bytes) -> bool:
        """Verify a signature using a public key with enhanced validation"""
        try:
            # Parameter validation
            if not signature or not data or not public_key_bytes:
                return False
                
            # Check key format
            if len(public_key_bytes) != 32:
                logger.warning("Invalid public key length")
                return False
                
            # Check signature format
            if len(signature) != 64:
                logger.warning("Invalid signature length")
                return False
                
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
            public_key.verify(signature, data)
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"Verification error: {str(e)}")
            return False

class CognitivePacket:
    # ... existing implementation ...
    
    def verify_vet_signature(self) -> bool:
        """Verify the packet signature with strict input validation"""
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
