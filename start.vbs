' start.vbs — launches Pymss Studio with NO visible cmd/console window.
' Double-click this file instead of start.bat.
Option Explicit
Dim oShell, oFSO, strBat
Set oShell = CreateObject("WScript.Shell")
Set oFSO   = CreateObject("Scripting.FileSystemObject")
strBat = oFSO.BuildPath(oFSO.GetParentFolderName(WScript.ScriptFullName), "start.bat")
' Tell start.bat it is already hidden, so it won't re-relaunch itself.
oShell.Environment("Process").Item("PSS_HIDDEN") = "1"
' Run style 0 = hidden window. False = do not wait for the batch to finish.
oShell.Run """" & strBat & """", 0, False
Set oShell = Nothing
Set oFSO   = Nothing
