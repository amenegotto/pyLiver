@echo off
FOR /F "usebackq tokens=*" %%i IN (slices.csv) DO (
     echo %%~i
     "C:\Program Files\MicroDicom\mDicom.exe" "%%~i"
     pause
)