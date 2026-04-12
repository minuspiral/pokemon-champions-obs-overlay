@echo off
echo === PokemonOverlay ビルド ===
echo.
echo [1] onedir版 (推奨、AV誤検知少ない、ZIPで配布)
echo [2] onefile版 (単体exe、便利だがAV誤検知されやすい)
echo.
set /p choice="選択 (1/2): "

if "%choice%"=="2" goto onefile

:onedir
pyinstaller --onedir --windowed --noupx --name "PokemonOverlay" ^
  --add-data "templates;templates" ^
  --hidden-import obsws_python ^
  --icon app_icon.ico ^
  --version-file version_info.txt ^
  overlay.py
echo.
echo ビルド完了: dist\PokemonOverlay\PokemonOverlay.exe
echo 配布時は dist\PokemonOverlay\ フォルダをzipで固めてください
goto end

:onefile
pyinstaller --onefile --windowed --noupx --name "PokemonOverlay" ^
  --add-data "templates;templates" ^
  --hidden-import obsws_python ^
  --icon app_icon.ico ^
  --version-file version_info.txt ^
  overlay.py
echo.
echo ビルド完了: dist\PokemonOverlay.exe

:end
pause
