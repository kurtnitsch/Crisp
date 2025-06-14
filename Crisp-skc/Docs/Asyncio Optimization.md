class SKCManager:
    # ... existing implementation ...
    
    async def process_packet(self, packet: CognitivePacket) -> bool:
        """Process packet with asyncio optimization"""
        if packet.hops > MAX_HOPS:
            logger.warning(f"Packet exceeded max hops: {packet.msg_id}")
            return False
            
        if packet.vet_signature and not packet.verify_vet_signature():
            logger.warning(f"Invalid signature on packet: {packet.msg_id}")
            return False
            
        handlers = self.packet_handlers.get(packet.msg_type, [])
        if not handlers:
            logger.debug(f"No handlers for message type: {packet.msg_type}")
            return False
        
        # Use gather with return_exceptions to prevent one failure blocking others
        results = await asyncio.gather(
            *(h(packet) for h in handlers),
            return_exceptions=True
        )
        
        # Log exceptions but don't propagate
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Handler error: {str(r)}", exc_info=True)
        
        return any(r is True for r in results if not isinstance(r, Exception))

class SmallPond:
    # ... existing implementation ...
    
    async def start(self, duration: float):
        """Start with optimized async processing"""
        # ... setup ...
        
        while self._running and time.monotonic() < end_time:
            # Process agents in batches to prevent event loop blocking
            batch_size = max(1, len(self.agents) // 10)
            for i in range(0, len(self.agents), batch_size):
                batch = self.agents[i:i+batch_size]
                tasks = [agent.tick() for agent in batch]
                await asyncio.gather(*tasks)
                
                # Yield control to event loop periodically
                if i % (batch_size * 2) == 0:
                    await asyncio.sleep(0)
            
            # ... timing control ...
