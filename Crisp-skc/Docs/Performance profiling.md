# crisp_skc.py
import cProfile
import pstats

class SmallPond:
    # ... existing implementation ...
    
    async def start(self, duration: float, profile: bool = False):
        """Start the pond simulation with optional profiling"""
        if self._running: 
            return
            
        self._running = True
        await self.kip_manager.start()
        
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()
            
        # ... simulation loop ...
        
        if profile:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)
            stats.dump_stats(f"smallpond_{self.pond_id}.prof")

# Example usage in demo_pond()
async def demo_pond():
    pond = SmallPond("performance-test")
    # ... setup ...
    await pond.start(duration=30, profile=True)
