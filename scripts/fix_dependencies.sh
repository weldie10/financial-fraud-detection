#!/bin/bash
# Script to fix dependency compatibility issues between scikit-learn and imbalanced-learn

echo "🔧 Fixing dependency compatibility issues..."
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Activating virtual environment..."
    source venv/bin/activate 2>/dev/null || {
        echo "❌ Virtual environment not found. Please activate it first:"
        echo "   source venv/bin/activate"
        exit 1
    }
fi

echo "📦 Checking current versions..."
pip list | grep -E "(scikit-learn|imbalanced-learn)" || echo "   (packages not installed)"

echo ""
echo "🔄 Updating packages to compatible versions..."
echo "   (scikit-learn <1.5.0 with imbalanced-learn <0.12.0)"
echo ""

# Uninstall conflicting packages
echo "   Uninstalling old versions..."
pip uninstall -y scikit-learn imbalanced-learn 2>/dev/null

# Install compatible versions
echo "   Installing compatible versions..."
pip install "scikit-learn>=1.3.0,<1.5.0"
pip install "imbalanced-learn>=0.11.0,<0.12.0"

echo ""
echo "✅ Dependencies updated!"
echo ""
echo "📊 Current versions:"
pip list | grep -E "(scikit-learn|imbalanced-learn)"

echo ""
echo "🚀 You can now run: python scripts/train_model.py"
