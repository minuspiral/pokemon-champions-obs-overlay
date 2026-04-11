@echo off
echo === PokemonOverlay exe ビルド ===
pyinstaller --onefile --windowed --name "PokemonOverlay" ^
  --add-data "templates;templates" ^
  --hidden-import obsws_python ^
  --console ^
  overlay.py
echo.
echo ビルド完了: dist\PokemonOverlay.exe
pause
