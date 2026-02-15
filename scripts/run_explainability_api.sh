#!/bin/bash
# Script to run the Explainability API

echo "🚀 Starting Explainability API..."
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "Activating virtual environment..."
    source venv/bin/activate 2>/dev/null || echo "Please activate venv manually: source venv/bin/activate"
fi

# Check if flask is installed
if ! command -v flask &> /dev/null && ! python -c "import flask" 2>/dev/null; then
    echo "❌ Flask not found. Installing..."
    pip install flask flask-cors
fi

# Navigate to project root
cd "$(dirname "$0")/.."

# Check if API file exists
if [ ! -f "api/explainability_api.py" ]; then
    echo "❌ API file not found at api/explainability_api.py"
    exit 1
fi

# Run API
echo "📡 API will be available at http://localhost:5000"
echo "📚 API documentation: api/README.md"
echo ""
echo "Press Ctrl+C to stop the API"
echo ""

python api/explainability_api.py
