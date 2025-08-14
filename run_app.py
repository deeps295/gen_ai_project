#!/usr/bin/env python3
"""
Budget Buddy - Fast Startup Script
This script runs the Budget Buddy app with optimized settings for better performance.
"""

import subprocess
import sys
import os

def main():
    """Run the Budget Buddy app with optimized settings"""
    print("🚀 Starting Budget Buddy - Personal Finance Assistant")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found in current directory")
        print("Please run this script from the deepika/deepika/ directory")
        sys.exit(1)
    
    # Optimized Streamlit settings for better performance
    env = os.environ.copy()
    env.update({
        "STREAMLIT_SERVER_MAX_UPLOAD_SIZE": "200",
        "STREAMLIT_SERVER_ENABLE_STATIC_SERVING": "true",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_SERVER_ENABLE_CORS": "false",
        "STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION": "false"
    })
    
    try:
        # Run Streamlit with optimized settings
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("✅ Starting app with optimized settings...")
        print("🌐 App will be available at: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the app")
        print("-" * 50)
        
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
