' Lanza Subtítulos-pro sin ventana de consola
' Simplemente haz doble clic en este archivo para abrir el programa.

Dim fso, shell, curDir, pythonwPath
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

' Obtener la carpeta donde está este VBS
curDir = fso.GetAbsolutePathName(".")
shell.CurrentDirectory = curDir

' Usar pythonw.exe del entorno virtual (evita depender del PATH)
pythonwPath = curDir & "\.venv\Scripts\pythonw.exe"

' Si no existe, probar pythonw.exe global
If Not fso.FileExists(pythonwPath) Then
    pythonwPath = "pythonw.exe"
End If

' 0 = ventana oculta, False = no esperar a que termine
shell.Run """" & pythonwPath & """ main.py", 0, False
