"""
Development environment setup script
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    requirements = [
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "plotly>=5.15.0",
        "scipy>=1.10.0",
        "requests>=2.28.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pytest>=7.4.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "jupyter>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

def create_directories():
    """Create necessary directories"""
    print("\nCreating project directories...")
    
    directories = [
        "data",
        "docs",
        "examples",
        "scripts",
        "tests",
        "notebooks",
        "output"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def setup_git_hooks():
    """Setup git hooks for code quality"""
    print("\nSetting up git hooks...")
    
    pre_commit_hook = """#!/bin/bash
# Run black formatter
black src/ tests/ examples/

# Run flake8 linter
flake8 src/ tests/ examples/ --max-line-length=100

# Run tests
pytest tests/ -v
"""
    
    hooks_dir = ".git/hooks"
    if os.path.exists(".git"):
        os.makedirs(hooks_dir, exist_ok=True)
        
        with open(os.path.join(hooks_dir, "pre-commit"), "w") as f:
            f.write(pre_commit_hook)
        
        # Make executable
        os.chmod(os.path.join(hooks_dir, "pre-commit"), 0o755)
        print("✓ Created pre-commit hook")
    else:
        print("⚠ Git repository not found, skipping git hooks setup")

def create_config_files():
    """Create configuration files"""
    print("\nCreating configuration files...")
    
    # pytest.ini
    pytest_config = """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short
"""
    
    with open("pytest.ini", "w") as f:
        f.write(pytest_config)
    print("✓ Created pytest.ini")
    
    # .flake8
    flake8_config = """[flake8]
max-line-length = 100
exclude = .git,__pycache__,build,dist
ignore = E203,W503
"""
    
    with open(".flake8", "w") as f:
        f.write(flake8_config)
    print("✓ Created .flake8")
    
    # pyproject.toml for black
    pyproject_config = """[tool.black]
line-length = 100
target-version = ['py38']
include = '\\.pyi?$'
extend-exclude = '''
/(
  # directories
  \\.eggs
  | \\.git
  | \\.hg
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | build
  | dist
)/
'''
"""
    
    with open("pyproject.toml", "w") as f:
        f.write(pyproject_config)
    print("✓ Created pyproject.toml")

def main():
    """Main setup function"""
    print("PM-Analyzer Development Environment Setup")
    print("=" * 45)
    
    install_requirements()
    create_directories()
    setup_git_hooks()
    create_config_files()
    
    print("\n" + "=" * 45)
    print("Development environment setup complete!")
    print("\nNext steps:")
    print("1. Run 'streamlit run main.py' to start the application")
    print("2. Run 'pytest tests/' to run the test suite")
    print("3. Run 'python examples/basic_usage.py' for usage examples")
    print("4. Check out the examples/ directory for more advanced usage")

if __name__ == "__main__":
    main()
