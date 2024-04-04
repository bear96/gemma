@echo off

rem Prompt the user to enter the directory path
set /p "dir_path=Enter the directory path containing .tex files: "

rem Check if the directory exists
if not exist "%dir_path%" (
    echo Directory does not exist.
    exit /b
)

rem Change to the specified directory
cd /d "%dir_path%"

rem Loop through all .tex files in the specified directory
for %%i in (*.tex) do (
    rem Extract the filename without extension
    set "filename=%%~ni"
    
    rem Run pdflatex to convert .tex to .pdf
    pdflatex "%%i"
    
    rem Check if PDF file was created successfully
    if exist "%filename%.pdf" (
        echo Conversion successful for "%%i"
    ) else (
        echo Conversion failed for "%%i"
    )
)
