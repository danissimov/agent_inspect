# PowerShell script to copy .eval files and rename them to .zip
# This script will:
# 1. Find all .eval files in the current directory
# 2. Create a copy of each file with .zip extension
# 3. Preserve the original .eval files

# Get all .eval files in the current directory
$evalFiles = Get-ChildItem -Filter "*.eval"

# Check if any .eval files were found
if ($evalFiles.Count -eq 0) {
    Write-Host "No .eval files found in the current directory."
    exit
}

# Process each .eval file
foreach ($file in $evalFiles) {
    # Create the new filename with .zip extension
    $newName = $file.FullName -replace '\.eval$', '.zip'
    
    # Copy the file with the new extension
    Copy-Item -Path $file.FullName -Destination $newName
    
    Write-Host "Copied and renamed: $($file.Name) -> $($file.BaseName).zip"
}

Write-Host "`nOperation completed. $($evalFiles.Count) files were processed." 