#!/usr/bin/env python3
"""
DTU Aqua Vision Test Runner

Run all tests or specific test categories for the DTU Aqua Vision project.
"""

import subprocess
import sys
import os
import argparse
import time

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n🧪 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    end_time = time.time()
    
    duration = end_time - start_time
    
    if result.returncode == 0:
        print(f"✅ {description} completed in {duration:.2f}s")
        return True
    else:
        print(f"❌ {description} failed after {duration:.2f}s")
        return False

def check_backend_running():
    """Check if backend server is running"""
    try:
        import requests
        response = requests.get("http://localhost:8000/models", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description="Run DTU Aqua Vision tests")
    parser.add_argument(
        "--category", 
        choices=["api", "detection", "websocket", "performance", "all"],
        default="all",
        help="Test category to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-backend-check",
        action="store_true",
        help="Skip backend availability check"
    )
    
    args = parser.parse_args()
    
    # Change to tests directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("🚀 DTU Aqua Vision Test Suite")
    print("=" * 50)
    
    # Check if backend is running (unless skipped)
    if not args.no_backend_check:
        print("🔍 Checking backend availability...")
        if not check_backend_running():
            print("⚠️  Backend server not detected at localhost:8000")
            print("   Please start the backend server first:")
            print("   cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
            print("\n   Or use --no-backend-check to skip this check")
            return 1
        else:
            print("✅ Backend server is running")
    
    # Build pytest command
    pytest_args = ["python", "-m", "pytest"]
    
    if args.verbose:
        pytest_args.extend(["-v", "-s"])
    
    # Add specific test files based on category
    test_files = []
    
    if args.category in ["api", "all"]:
        test_files.append("backend/test_api.py")
    
    if args.category in ["detection", "all"]:
        test_files.append("backend/test_detection.py")
    
    if args.category in ["websocket", "all"]:
        test_files.append("performance/test_websocket.py")
    
    if args.category in ["performance", "all"]:
        # Run all performance tests
        if "performance/test_websocket.py" not in test_files:
            test_files.append("performance/test_websocket.py")
    
    # Add test files to command
    pytest_args.extend(test_files)
    
    # Run tests
    success = True
    
    if test_files:
        success = run_command(
            pytest_args,
            f"Running {args.category} tests"
        )
    else:
        print(f"❌ No tests found for category: {args.category}")
        return 1
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 