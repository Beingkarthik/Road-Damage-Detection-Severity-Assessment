@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install -r requirements.txt
if errorlevel 1 pause & exit /b 1
python -m streamlit run app.py --server.port 8501
pause
