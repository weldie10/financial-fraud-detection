#!/bin/bash
# Code Quality Check Script
# Run all quality checks locally before committing

set -e  # Exit on error

echo "🔍 Running Code Quality Checks..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}⚠️  Warning: Virtual environment not activated${NC}"
    echo "Activate with: source venv/bin/activate"
    echo ""
fi

# Check if dependencies are installed
if ! command -v black &> /dev/null; then
    echo -e "${YELLOW}⚠️  Installing development dependencies...${NC}"
    pip install -r requirements-dev.txt
fi

echo "1️⃣  Running Black (code formatting)..."
black --check --diff src/ tests/ || {
    echo -e "${RED}❌ Black check failed. Run 'black src/ tests/' to fix.${NC}"
    exit 1
}
echo -e "${GREEN}✅ Black check passed${NC}"
echo ""

echo "2️⃣  Running isort (import sorting)..."
isort --check-only --diff src/ tests/ || {
    echo -e "${RED}❌ isort check failed. Run 'isort src/ tests/' to fix.${NC}"
    exit 1
}
echo -e "${GREEN}✅ isort check passed${NC}"
echo ""

echo "3️⃣  Running Flake8 (code quality)..."
flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics || {
    echo -e "${RED}❌ Flake8 found critical errors${NC}"
    exit 1
}
flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
echo -e "${GREEN}✅ Flake8 check passed${NC}"
echo ""

echo "4️⃣  Running MyPy (type checking)..."
mypy src/ --ignore-missing-imports --no-strict-optional --warn-redundant-casts --warn-unused-ignores || {
    echo -e "${YELLOW}⚠️  MyPy found type issues (non-blocking)${NC}"
}
echo -e "${GREEN}✅ MyPy check completed${NC}"
echo ""

echo "5️⃣  Running Pytest (tests)..."
pytest tests/ -v --tb=short --cov=src --cov-report=term-missing || {
    echo -e "${RED}❌ Tests failed${NC}"
    exit 1
}
echo -e "${GREEN}✅ All tests passed${NC}"
echo ""

echo "6️⃣  Running Safety (dependency vulnerabilities)..."
safety check --file requirements.txt || {
    echo -e "${YELLOW}⚠️  Safety found vulnerabilities (non-blocking)${NC}"
}
echo -e "${GREEN}✅ Safety check completed${NC}"
echo ""

echo "7️⃣  Running Bandit (security linting)..."
bandit -r src/ -ll || {
    echo -e "${YELLOW}⚠️  Bandit found security issues (non-blocking)${NC}"
}
echo -e "${GREEN}✅ Bandit check completed${NC}"
echo ""

echo -e "${GREEN}✅ All quality checks completed successfully!${NC}"
echo "Code is ready to commit and push."
