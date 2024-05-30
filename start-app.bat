@echo off
IF NOT EXIST .env (
    python -m venv .env
    cd .env\Scripts
    call activate.bat
    cd ../..
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
) ELSE (
    cd .env\Scripts
    call activate.bat
    cd ../..
)

python.exe app.py