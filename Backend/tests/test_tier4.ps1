
$baseUrl = "http://127.0.0.1:8000/api"
$ErrorActionPreference = "Stop"
Start-Transcript -Path "$PWD\test_tier4.log" -Force

$csvContent = @"
id,name,age,salary,join_date
1,John Doe,30,50000,2020-01-01
2,Jane Smith,,60000,2019-05-15
3,Bob Jones,45,75000,
4,John Doe,30,50000,2020-01-01
5,Alice Brown,28,NaN,2021-03-10
"@

# Write CSV to a temporary file
$tempCsv = [System.IO.Path]::GetTempFileName() + ".csv"
$csvContent | Out-File -Encoding UTF8 $tempCsv
Write-Host "Temp CSV created at: $tempCsv"

# Function to upload file using multipart/form-data via curl if Invoke-RestMethod fails with complexity
function Upload-File {
    param($filePath)
    $uri = "$baseUrl/upload"
    $boundary = [System.Guid]::NewGuid().ToString()
    $LF = "`r`n"
    
    $fileBytes = [System.IO.File]::ReadAllBytes($filePath)
    $fileHeader = "--$boundary",
                  "Content-Disposition: form-data; name=`"file`"; filename=`"test_data.csv`"",
                  "Content-Type: text/csv",
                  "",
                  "" -join $LF
    
    $fileFooter = "$LF--$boundary--$LF"
    
    $bodyBytes = [System.Text.Encoding]::ASCII.GetBytes($fileHeader) + $fileBytes + [System.Text.Encoding]::ASCII.GetBytes($fileFooter)
    
    try {
        $wr = [System.Net.WebRequest]::Create($uri)
        $wr.Method = "POST"
        $wr.ContentType = "multipart/form-data; boundary=$boundary"
        $wr.ContentLength = $bodyBytes.Length
        $stream = $wr.GetRequestStream()
        $stream.Write($bodyBytes, 0, $bodyBytes.Length)
        $stream.Close()
        
        $resp = $wr.GetResponse()
        $reader = New-Object System.IO.StreamReader($resp.GetResponseStream())
        $json = $reader.ReadToEnd()
        $reader.Close()
        return $json | ConvertFrom-Json
    } catch {
        Write-Error "Upload failed: $_"
        throw
    }
}


Write-Host "1. Uploading Test CSV..."
# Using simpler Invoke-RestMethod for upload if possible, but PowerShell 5.1 is finicky with multipart
# Let's try the function method which is reliable
try {
    $response = Upload-File -filePath $tempCsv
    $taskId = $response.task_id
    Write-Host "   Task ID: $taskId"
} catch {
    Write-Error "Upload failed. Is server running?"
    exit
}

Write-Host "2. Waiting for processing..."
$startWait = Get-Date
while ((Get-Date) -lt $startWait.AddSeconds(60)) {
    Start-Sleep -Seconds 2
    try {
        $status = Invoke-RestMethod -Uri "$baseUrl/status/$taskId" -Method Get
        Write-Host "   Status: $($status.status) - $($status.progress)%"
        
        if ($status.status -eq "completed") { 
            Write-Host "   Processing Complete!"
            break 
        }
        if ($status.status -eq "waiting_for_user") {
            Write-Host "   Job waiting for approval. Approving defaults..."
            try {
                $approveBody = @{ rules = @{} } | ConvertTo-Json
                Invoke-RestMethod -Uri "$baseUrl/jobs/$taskId/analyze" -Method Post -Body $approveBody -ContentType "application/json"
                Write-Host "   Approved! Waiting for analysis..."
            } catch {
                Write-Error "Failed to approve job: $_"
                exit
            }
        }
        if ($status.status -eq "failed") { 
            Write-Error "Job Failed: $($status.error)"
            exit
        }
    } catch {
        Write-Warning "Status check failed, retrying..."
    }
}

Write-Host "3. getting Comparison Report..."
try {
    $comparison = Invoke-RestMethod -Uri "$baseUrl/jobs/$taskId/comparison" -Method Get
    if ($comparison) {
        Write-Host "   Success! Comparison Summary:"
        # Accessing nested properties can fail if null, verify
        if ($comparison.summary) {
            Write-Host "   Completeness: $($comparison.summary.completeness.before)% -> $($comparison.summary.completeness.after)%"
            Write-Host "   Uniqueness:   $($comparison.summary.uniqueness.before)% -> $($comparison.summary.uniqueness.after)%"
        }
        
        if ($comparison.columns -and $comparison.columns.age) {
            $ageMetrics = $comparison.columns.age.metrics | Where-Object { $_.metric -eq 'missing_count' }
            Write-Host "   Age Missing: $($ageMetrics.before) -> $($ageMetrics.after)"
        }
    } else {
        Write-Error "Comparison report returned empty/null"
    }
} catch {
    Write-Error "Failed to get comparison report: $_"
}

Write-Host "4. Downloading PDF Report..."
try {
    $pdfUrl = "$baseUrl/jobs/$taskId/report"
    # Download to current directory
    $outFile = Join-Path $PWD "test_report.pdf"
    Invoke-RestMethod -Uri $pdfUrl -Method Get -OutFile $outFile
    
    if (Test-Path $outFile) {
        $fileInfo = Get-Item $outFile
        if ($fileInfo.Length -gt 1000) {
            Write-Host "   Success! PDF downloaded ($($fileInfo.Length) bytes)"
        } else {
            Write-Warning "   PDF downloaded but seems too small ($($fileInfo.Length) bytes). Check content."
        }
    } else {
        Write-Error "PDF file not found after download attempt."
    }
} catch {
    Write-Error "Failed to download PDF: $_"
}

# Cleanup
Remove-Item $tempCsv -ErrorAction SilentlyContinue
