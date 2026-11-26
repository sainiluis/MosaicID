@echo off
REM Empaqueta MosaicID como carpeta portable (dist\MosaicID\MosaicID.exe).
REM Requiere el entorno virtual con requirements.txt instalado.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No existe .venv. Crealo con:  python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
  exit /b 1
)
if not exist "models\face_detection_yunet_2023mar.onnx" ( echo Falta models\face_detection_yunet_2023mar.onnx & exit /b 1 )
if not exist "models\face_recognition_sface_2021dec.onnx" ( echo Falta models\face_recognition_sface_2021dec.onnx & exit /b 1 )
echo == Tests ==
".venv\Scripts\python.exe" -m pytest -q tests || exit /b 1
echo == Resguardo de datos de dist (PyInstaller borra la carpeta) ==
if exist "_keep_dist" rmdir /s /q "_keep_dist"
mkdir "_keep_dist"
for %%D in (data config cache logs) do if exist "dist\MosaicID\%%D" move "dist\MosaicID\%%D" "_keep_dist\%%D" >nul
REM Si alguna carpeta no se pudo mover (archivo en uso), PyInstaller la borraria: abortar sin tocar nada.
for %%D in (data config cache logs) do if exist "dist\MosaicID\%%D" goto :keepfail
echo == PyInstaller ==
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean mosaicid.spec
set ERR=%ERRORLEVEL%
if not exist "dist\MosaicID" mkdir "dist\MosaicID"
for %%D in (data config cache logs) do if exist "_keep_dist\%%D" move "_keep_dist\%%D" "dist\MosaicID\%%D" >nul
if %ERR% neq 0 exit /b %ERR%
if exist "galeria_incluida" robocopy "galeria_incluida" "dist\MosaicID\galeria_incluida" /E /NFL /NDL /NJH /NJS >nul
echo.
echo Listo: dist\MosaicID\MosaicID.exe
echo Los datos (data, config, cache, logs) se crean junto al .exe en la primera ejecucion.
for %%D in (data config cache logs) do if exist "_keep_dist\%%D" echo ATENCION: _keep_dist\%%D no se pudo devolver a dist\MosaicID
endlocal
exit /b 0

:keepfail
echo ERROR: no se pudo resguardar una carpeta de dist\MosaicID (archivo en uso). Cerra MosaicID y reintenta.
for %%D in (data config cache logs) do if exist "_keep_dist\%%D" move "_keep_dist\%%D" "dist\MosaicID\%%D" >nul
endlocal
exit /b 1
