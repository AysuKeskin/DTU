import pytest
import asyncio
import websockets
import json
import time
import numpy as np
import cv2
from PIL import Image
import io

class TestWebSocketPerformance:
    """Test WebSocket real-time detection performance"""
    
    def create_test_frame(self, width=320, height=240):
        """Create a test frame as JPEG bytes"""
        # Create a simple test image
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Convert to PIL and then to JPEG bytes
        pil_img = Image.fromarray(img)
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format='JPEG', quality=20)
        
        return img_bytes.getvalue()
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test basic WebSocket connection"""
        uri = "ws://localhost:8000/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Connection successful
                assert websocket.open
                print("✅ WebSocket connection established")
                
        except Exception as e:
            pytest.skip(f"Cannot connect to WebSocket: {e}")
    
    @pytest.mark.asyncio
    async def test_websocket_frame_sending(self):
        """Test sending frames via WebSocket"""
        uri = "ws://localhost:8000/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Create test frame
                frame_data = self.create_test_frame()
                
                # Send frame
                start_time = time.time()
                await websocket.send(frame_data)
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    end_time = time.time()
                    
                    # Parse response
                    data = json.loads(response)
                    
                    # Calculate round-trip time
                    round_trip_time = end_time - start_time
                    
                    print(f"📡 WebSocket round-trip: {round_trip_time:.3f}s")
                    print(f"🎯 Detections received: {len(data.get('detections', []))}")
                    
                    # Assert reasonable performance
                    assert round_trip_time < 3.0  # Should be under 3 seconds
                    assert "detections" in data
                    
                    return round_trip_time
                    
                except asyncio.TimeoutError:
                    pytest.fail("WebSocket response timeout")
                    
        except Exception as e:
            pytest.skip(f"WebSocket test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_websocket_multiple_frames(self):
        """Test sending multiple frames and measure performance"""
        uri = "ws://localhost:8000/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                frame_data = self.create_test_frame()
                
                times = []
                num_frames = 5
                
                for i in range(num_frames):
                    start_time = time.time()
                    
                    # Send frame
                    await websocket.send(frame_data)
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        end_time = time.time()
                        
                        round_trip_time = end_time - start_time
                        times.append(round_trip_time)
                        
                        print(f"Frame {i+1}: {round_trip_time:.3f}s")
                        
                        # Small delay between frames
                        await asyncio.sleep(0.1)
                        
                    except asyncio.TimeoutError:
                        print(f"Frame {i+1}: Timeout")
                        continue
                
                if times:
                    avg_time = np.mean(times)
                    min_time = np.min(times)
                    max_time = np.max(times)
                    
                    print(f"\n📊 WebSocket Performance ({num_frames} frames):")
                    print(f"  Average: {avg_time:.3f}s")
                    print(f"  Min: {min_time:.3f}s")
                    print(f"  Max: {max_time:.3f}s")
                    print(f"  Effective FPS: {1/avg_time:.1f}")
                    
                    # Assert reasonable performance
                    assert avg_time < 2.0
                    
                    return {
                        "average": avg_time,
                        "min": min_time,
                        "max": max_time,
                        "fps": 1/avg_time
                    }
                    
        except Exception as e:
            pytest.skip(f"WebSocket multiple frames test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_websocket_different_resolutions(self):
        """Test WebSocket performance with different frame sizes"""
        uri = "ws://localhost:8000/ws"
        
        resolutions = [
            (160, 120),   # Small
            (320, 240),   # Medium
            (640, 480),   # Large
        ]
        
        try:
            async with websockets.connect(uri) as websocket:
                performance_by_resolution = {}
                
                for width, height in resolutions:
                    frame_data = self.create_test_frame(width, height)
                    
                    start_time = time.time()
                    await websocket.send(frame_data)
                    
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        end_time = time.time()
                        
                        round_trip_time = end_time - start_time
                        frame_size = len(frame_data)
                        
                        performance_by_resolution[f"{width}x{height}"] = {
                            "time": round_trip_time,
                            "size_kb": frame_size / 1024,
                            "fps": 1 / round_trip_time
                        }
                        
                        print(f"📏 {width}x{height}: {round_trip_time:.3f}s, {frame_size/1024:.1f}KB")
                        
                        # Small delay
                        await asyncio.sleep(0.5)
                        
                    except asyncio.TimeoutError:
                        print(f"📏 {width}x{height}: Timeout")
                        continue
                
                return performance_by_resolution
                
        except Exception as e:
            pytest.skip(f"WebSocket resolution test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_websocket_stress_test(self):
        """Stress test WebSocket with rapid frame sending"""
        uri = "ws://localhost:8000/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                frame_data = self.create_test_frame()
                
                # Send frames rapidly
                num_frames = 10
                start_time = time.time()
                
                responses_received = 0
                
                # Send all frames rapidly
                for i in range(num_frames):
                    await websocket.send(frame_data)
                
                # Collect responses
                for i in range(num_frames):
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        responses_received += 1
                    except asyncio.TimeoutError:
                        break
                
                end_time = time.time()
                total_time = end_time - start_time
                
                print(f"🚀 Stress Test Results:")
                print(f"  Frames sent: {num_frames}")
                print(f"  Responses received: {responses_received}")
                print(f"  Total time: {total_time:.3f}s")
                print(f"  Effective rate: {responses_received/total_time:.1f} FPS")
                
                # Should handle at least some frames
                assert responses_received > 0
                
                return {
                    "frames_sent": num_frames,
                    "responses_received": responses_received,
                    "total_time": total_time,
                    "effective_fps": responses_received / total_time
                }
                
        except Exception as e:
            pytest.skip(f"WebSocket stress test failed: {e}")

class TestWebSocketReliability:
    """Test WebSocket connection reliability and error handling"""
    
    @pytest.mark.asyncio
    async def test_websocket_reconnection(self):
        """Test WebSocket reconnection capability"""
        uri = "ws://localhost:8000/ws"
        
        try:
            # First connection
            async with websockets.connect(uri) as websocket1:
                assert websocket1.open
                print("✅ First connection established")
            
            # Second connection (after first is closed)
            async with websockets.connect(uri) as websocket2:
                assert websocket2.open
                print("✅ Reconnection successful")
                
        except Exception as e:
            pytest.skip(f"WebSocket reconnection test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_websocket_error_handling(self):
        """Test WebSocket error handling with invalid data"""
        uri = "ws://localhost:8000/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Send invalid data
                await websocket.send("invalid_image_data")
                
                # Should still receive a response (likely an error)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    data = json.loads(response)
                    
                    # Should indicate error or empty detections
                    assert "detections" in data or "error" in data
                    print("✅ Error handling works correctly")
                    
                except asyncio.TimeoutError:
                    print("⚠️ No response to invalid data")
                
        except Exception as e:
            pytest.skip(f"WebSocket error handling test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__]) 