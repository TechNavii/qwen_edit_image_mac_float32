#!/bin/bash

echo "🎨 Qwen Image Editor - Starting Application"
echo "=========================================="

if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
    
    echo "📥 Installing dependencies (first time setup)..."
    pip install -q -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "✅ Using existing virtual environment"
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
    
    # Optional: Check if requirements have changed
    echo "🔍 Checking dependencies..."
    pip install -q --dry-run -r requirements.txt 2>&1 | grep -q "Would install" && {
        echo "📥 Updating dependencies..."
        pip install -q -r requirements.txt
        echo "✅ Dependencies updated"
    } || {
        echo "✅ All dependencies are up to date"
    }
fi

echo "🚀 Starting Gradio interface..."
echo "=========================================="
echo "📍 Local URL: http://localhost:7860"
echo "🛑 Press Ctrl+C to stop the server"
echo "=========================================="
python app.py
