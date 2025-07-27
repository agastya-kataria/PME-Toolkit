"""
Simple script to run the PM-Analyzer application
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'pandas', 'numpy', 'plotly', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        print("All packages installed successfully!")

def main():
    """Main function to run the app"""
    print("PM-Analyzer: Private Markets Portfolio Analyzer")
    print("=" * 50)
    
    # Check and install requirements
    check_requirements()
    
    # Run the Streamlit app
    print("Starting Streamlit application...")
    print("Open your browser and go to: http://localhost:8501")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error running application: {e}")

if __name__ == "__main__":
    main()
