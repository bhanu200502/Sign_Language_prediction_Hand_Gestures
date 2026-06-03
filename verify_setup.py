#!/usr/bin/env python3
"""
🎯 Model Evaluation Setup Verification Script
Checks all required files and dependencies for model evaluation

Run: python verify_setup.py
"""

import os
import sys

def check_file_exists(path, description):
    """Check if a required file exists"""
    if os.path.exists(path):
        print(f"  ✅ {description}")
        return True
    else:
        print(f"  ❌ {description} NOT FOUND: {path}")
        return False

def check_directory_exists(path, description):
    """Check if a required directory exists"""
    if os.path.isdir(path):
        print(f"  ✅ {description}")
        return True
    else:
        print(f"  ❌ {description} NOT FOUND: {path}")
        return False

def check_imports():
    """Check if all required packages are installed"""
    required_packages = {
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'streamlit': 'Streamlit',
        'mediapipe': 'MediaPipe',
    }
    
    print("\n📦 Checking Python Packages:")
    all_installed = True
    
    for module_name, display_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} NOT INSTALLED")
            all_installed = False
    
    return all_installed

def main():
    """Main verification routine"""
    print("\n" + "="*60)
    print("🎯 MODEL EVALUATION SETUP VERIFICATION")
    print("="*60)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Check required files
    print("\n📁 Checking Required Files:")
    
    checks = [
        (os.path.join(BASE_DIR, "src", "model_evaluation.py"), "Evaluation Script"),
        (os.path.join(BASE_DIR, "src", "models", "sign_model.h5"), "Trained Model"),
        (os.path.join(BASE_DIR, "src", "data", "X.npy"), "Feature Data (X.npy)"),
        (os.path.join(BASE_DIR, "src", "data", "y.npy"), "Labels Data (y.npy)"),
        (os.path.join(BASE_DIR, "src", "data", "labels.npy"), "Class Labels"),
        (os.path.join(BASE_DIR, "app.py"), "Streamlit App"),
        (os.path.join(BASE_DIR, "EVALUATION_GUIDE.md"), "Evaluation Guide"),
        (os.path.join(BASE_DIR, "EVALUATION_QUICK_REF.md"), "Quick Reference"),
    ]
    
    all_files_exist = True
    for file_path, description in checks:
        if not check_file_exists(file_path, description):
            all_files_exist = False
    
    # Check required directories
    print("\n📂 Checking Required Directories:")
    
    dir_checks = [
        (os.path.join(BASE_DIR, "src", "models"), "Models Directory"),
        (os.path.join(BASE_DIR, "src", "data"), "Data Directory"),
    ]
    
    all_dirs_exist = True
    for dir_path, description in dir_checks:
        if not check_directory_exists(dir_path, description):
            all_dirs_exist = False
    
    # Check imports
    all_imports = check_imports()
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    status_file = "✅" if all_files_exist else "❌"
    status_dir = "✅" if all_dirs_exist else "❌"
    status_pkg = "✅" if all_imports else "❌"
    
    print(f"{status_file} Files Required")
    print(f"{status_dir} Directories Required")
    print(f"{status_pkg} Python Packages")
    
    if all_files_exist and all_dirs_exist and all_imports:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\n✨ You're ready to generate metrics:")
        print("   Run: python src/model_evaluation.py")
        print("\nThen view in Streamlit:")
        print("   Run: streamlit run app.py")
        print("   Go to: 📈 Model Metrics tab")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\n🔧 Next steps:")
        
        if not all_files_exist:
            print("\n  1. Ensure model is trained:")
            print("     python src/model_training.py")
            print("\n  2. Ensure data preprocessing is complete:")
            print("     python src/data_preprocessing.py")
        
        if not all_imports:
            print("\n  3. Install missing packages:")
            print("     pip install -r requirements.txt")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
