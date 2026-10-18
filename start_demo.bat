   @echo off
   cd /d "%~dp0"
   call venv\Scripts\activate
   start "ChipPal server" cmd /k python -m uvicorn main:app --host 0.0.0.0 --port 8000
   timeout /t 30
   start "Tunnel" cmd /k cloudflared tunnel --url http://localhost:8000