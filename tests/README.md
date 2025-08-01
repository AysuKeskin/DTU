# DTU Aqua Vision Test Suite

Comprehensive testing framework for the DTU Aqua Vision real-time object detection system.

## 📁 Test Structure

```
tests/
├── backend/               # Backend API tests
│   ├── test_api.py       # API endpoint testing
│   └── test_detection.py # Detection accuracy & performance
├── performance/          # Performance benchmarks
│   └── test_websocket.py # WebSocket real-time performance
├── requirements.txt      # Test dependencies
├── run_tests.py         # Test runner script
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
cd tests
pip install -r requirements.txt
```

### 2. Start Backend Server

Make sure your backend is running:
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Run Tests

**Run all tests:**
```bash
python run_tests.py
```

**Run specific test categories:**
```bash
python run_tests.py --category api          # API tests only
python run_tests.py --category detection    # Detection tests only
python run_tests.py --category websocket    # WebSocket tests only
python run_tests.py --category performance  # Performance tests only
```

**Verbose output:**
```bash
python run_tests.py --verbose
```

## 🧪 Test Categories

### Backend API Tests (`test_api.py`)

Tests core API functionality:
- ✅ Model loading and switching
- ✅ Endpoint availability
- ✅ Model management
- ✅ Error handling

**Key Tests:**
- `test_get_models()` - Verify available models
- `test_load_model()` - Test model loading speed
- `test_model_switching_speed()` - Measure switching performance

### Detection Tests (`test_detection.py`)

Tests detection accuracy and performance:
- ✅ Detection accuracy with test images
- ✅ Performance with different image sizes
- ✅ Model comparison benchmarks
- ✅ Consistency testing

**Key Tests:**
- `test_detection_speed_benchmark()` - Speed benchmarking
- `test_model_comparison()` - Compare model performance
- `test_detection_performance_different_sizes()` - Size vs speed

### WebSocket Performance Tests (`test_websocket.py`)

Tests real-time detection performance:
- ✅ WebSocket connection reliability
- ✅ Frame processing speed
- ✅ Round-trip time measurement
- ✅ Stress testing
- ✅ Different resolution performance

**Key Tests:**
- `test_websocket_multiple_frames()` - Multi-frame performance
- `test_websocket_different_resolutions()` - Resolution impact
- `test_websocket_stress_test()` - System limits

## 📊 Performance Benchmarks

The tests will output performance metrics like:

```
📊 WebSocket Performance (5 frames):
  Average: 0.245s
  Min: 0.198s
  Max: 0.312s
  Effective FPS: 4.1

Model Performance Comparison:
  yolov8s-seg.pt: 0.234s, 3 detections
  yolov8n-seg.pt: 0.156s, 2 detections
```

## 🎯 Performance Optimization Testing

Perfect for testing your optimizations:

1. **Baseline Measurement:**
   ```bash
   python run_tests.py --category performance
   ```

2. **Make Optimizations** (change model, resolution, etc.)

3. **Compare Results:**
   ```bash
   python run_tests.py --category performance
   ```

## 🔧 Adding New Tests

### Backend Tests
Add tests to `backend/test_*.py`:
```python
def test_new_feature(self):
    response = client.get("/new-endpoint")
    assert response.status_code == 200
```

### Performance Tests
Add benchmarks to `performance/test_*.py`:
```python
@pytest.mark.asyncio
async def test_new_performance_metric(self):
    start_time = time.time()
    # ... test code ...
    end_time = time.time()
    assert (end_time - start_time) < threshold
```

## 📋 Test Results Interpretation

### ✅ Good Performance Indicators
- **WebSocket round-trip:** < 300ms
- **Detection speed:** < 200ms
- **Model loading:** < 10s
- **API response:** < 100ms

### ⚠️ Performance Warnings
- **WebSocket round-trip:** > 500ms
- **Detection speed:** > 500ms
- **Frame drops:** > 10%
- **Memory usage:** Growing over time

### ❌ Critical Issues
- **WebSocket timeout:** > 3s
- **Detection failure:** Any crashes
- **Model loading failure:** Cannot load models
- **API unavailable:** Connection refused

## 🐛 Troubleshooting

### Backend Not Running
```
⚠️ Backend server not detected at localhost:8000
```
**Solution:** Start backend server first

### WebSocket Connection Failed
```
Cannot connect to WebSocket: [Errno 61] Connection refused
```
**Solution:** Ensure backend WebSocket endpoint is working

### Import Errors
```
ModuleNotFoundError: No module named 'pytest'
```
**Solution:** Install test dependencies: `pip install -r requirements.txt`

### Performance Degradation
If tests show performance regression:
1. Check system resources (CPU, GPU, memory)
2. Verify model versions
3. Compare with baseline measurements
4. Check for memory leaks

## 🎯 Next Steps

After setting up tests, you can:
1. **Benchmark current performance**
2. **Implement optimizations** (nano model, lower resolution)
3. **Measure improvements**
4. **Set up CI/CD** for automated testing
5. **Add more test cases** as you develop new features

Happy testing! 🧪✨ 