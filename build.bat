@echo off
echo === PokemonOverlay exe ビルド ===
pyinstaller --onefile --windowed --noupx --name "PokemonOverlay" ^
  --add-data "templates;templates" ^
  --hidden-import obsws_python ^
  --icon app_icon.ico ^
  --version-file version_info.txt ^
  overlay.py
echo.
echo ビルド完了: dist\PokemonOverlay.exe
pause
