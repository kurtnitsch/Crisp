class SmallPond:
    # ... existing implementation ...
    
    async def stop(self):
        """Stop the pond simulation and clean up resources"""
        if not self._running:
            return
            
        self._running = False
        
        # Clean up agents
        for agent in self.agents:
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()
        
        # Clean up KIP manager
        await self.kip_manager.stop()
        
        # Clear spatial grid
        self._spatial_grid.clear()
        
        # Release memory pools
        if hasattr(self, 'memory_pool'):
            self.memory_pool.clear()
        
        logger.info(f"SmallPond {self.pond_id} stopped and resources cleaned")

class PondAgent:
    # ... existing implementation ...
    
    async def cleanup(self):
        """Release resources held by the agent"""
        # Clear large data structures
        self.knowledge.clear()
        self.known_claims.clear()
        
        # Reset cryptographic material
        if hasattr(self.vetkey, 'reset'):
            self.vetkey.reset()
        
        # Release any network connections
        if hasattr(self, '_network_session'):
            await self._network_session.close()

class MemoryPool:
    # ... existing implementation ...
    
    def clear(self):
        """Release all buffers in the pool"""
        self._pool = []
        self._allocation_count = 0
        self._reuse_count = 0
