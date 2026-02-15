#!/bin/bash
# Script to run the Fraud Detection Dashboard

echo "🛡️  Starting Fraud Detection Dashboard..."
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "Activating virtual environment..."
    source venv/bin/activate 2>/dev/null || echo "Please activate venv manually: source venv/bin/activate"
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit
fi

# Navigate to project root
cd "$(dirname "$0")/.."

# Check if dashboard app exists
if [ ! -f "dashboard/app.py" ]; then
    echo "❌ Dashboard app not found at dashboard/app.py"
    exit 1
fi

# Run dashboard
echo "🚀 Launching dashboard..."
echo "📊 Dashboard will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

streamlit run dashboard/app.py
