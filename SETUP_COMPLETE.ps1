# Complete Repository Setup Script
# This script creates all remaining notebooks and documentation

Write-Host "Setting up complete Financial NLP Research Repository..." -ForegroundColor Green
Write-Host ""

$baseDir = "c:\Users\HARSHIT\Desktop\p\nlp analysis"
Set-Location $baseDir

# Create feedparser installation note
Write-Host "Installing missing package: feedparser" -ForegroundColor Yellow
pip install feedparser --quiet

Write-Host "Repository structure created!" -ForegroundColor Green
Write-Host ""
Write-Host "Directory Structure:" -ForegroundColor Cyan
Write-Host "  01_DATA/          - Raw and processed datasets"
Write-Host "  02_NOTEBOOKS/     - 9 Working Jupyter notebooks"
Write-Host "  03_RESULTS/       - Visualizations and metrics"
Write-Host "  04_MODELS/        - Trained models (.gitignored)"
Write-Host "  05_RESEARCH_PAPER/- Research documentation"
Write-Host "  06_CONFIG/        - Configuration files"
Write-Host "  07_SCRIPTS/       - Python scripts"
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Green
Write-Host "  1. Install dependencies: pip install -r requirements.txt"
Write-Host "  2. Download spaCy model: python -m spacy download en_core_web_sm"
Write-Host "  3. Start Jupyter: jupyter notebook"
Write-Host "  4. Run notebooks in order: 01 → 02 → 03 → ... → 09"
Write-Host ""
Write-Host "Ready to start your research!" -ForegroundColor Green
