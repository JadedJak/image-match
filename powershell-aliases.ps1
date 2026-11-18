# image-match convenience commands for PowerShell.
#
# Install: dot-source this file from your $PROFILE so it loads in every new
# session, e.g. add this one line to $PROFILE (see README "Setup" for the
# full install steps):
#
#     . D:\image-match\powershell-aliases.ps1
#
# Then open a new PowerShell window to pick it up (`. $PROFILE` in the
# same window has not reliably reloaded these functions).

$ImageMatchDir = "D:\image-match"

function indexer {
    <#
    .SYNOPSIS
    Run the indexer for one category, or both sequentially.
    .EXAMPLE
    indexer general
    indexer personal
    indexer all
    #>
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [ValidateSet("general", "personal", "all")]
        [string]$Category
    )
    Push-Location $ImageMatchDir
    try {
        if ($Category -eq "all") {
            # README: don't run both concurrently if their source volumes
            # share a physical disk -- sequential is always safe.
            docker compose run --rm imgmatch-general python -m app.indexer
            docker compose run --rm imgmatch-personal python -m app.indexer
        } else {
            docker compose run --rm "imgmatch-$Category" python -m app.indexer
        }
    } finally {
        Pop-Location
    }
}

function matchfind {
    <#
    .SYNOPSIS
    Bulk-match low-quality images (dropped in the category's match-input
    folder) against the catalog, copying the best higher-res match for each
    into the match-output folder. See README "Bulk matching low-quality
    images against the catalog".
    .EXAMPLE
    matchfind general
    matchfind personal
    #>
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [ValidateSet("general", "personal")]
        [string]$Category
    )
    Push-Location $ImageMatchDir
    try {
        docker compose run --rm "imgmatch-$Category" python -m app.bulk_match
    } finally {
        Pop-Location
    }
}

function _ImageMatchWaitReady {
    # Polls the service's log until the UI reports it's listening (NiceGUI
    # "NiceGUI ready", Gradio "Running on local URL"). Returns $false if the
    # container stops or it isn't ready within the timeout.
    param([string]$Service, [int]$TimeoutSec = 300)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $log = docker compose logs --no-log-prefix $Service 2>&1 | Out-String
        if ($log -match "NiceGUI ready|Running on local URL") { return $true }
        if (-not (docker compose ps -q --status running $Service)) { return $false }
        Start-Sleep -Seconds 3
    }
    return $false
}

function _ImageMatchStartBoth {
    # One at a time: started together, both containers load the ~4GB CLIP
    # model at the same moment on a host whose GPU/RAM is shared with
    # Ollama and Immich, and both would often fail. Starting personal only
    # after general reports ready has been reliable.
    param([string[]]$UpArgs)
    foreach ($svc in "imgmatch-general", "imgmatch-personal") {
        Write-Host "== starting $svc"
        docker compose up -d @UpArgs $svc
        if (-not (_ImageMatchWaitReady $svc)) {
            Write-Host "!! $svc did not come up -- check: image-match logs $($svc -replace 'imgmatch-','')" -ForegroundColor Red
            return
        }
        Write-Host "== $svc ready"
    }
}

function image-match {
    <#
    .SYNOPSIS
    Manage the image-match UI containers.
    .EXAMPLE
    image-match                     # up -d (default)
    image-match up
    image-match down
    image-match build
    image-match restart general     # restart just imgmatch-general
    image-match restart             # restart both
    image-match logs personal       # follow imgmatch-personal logs
    image-match stop personal       # stop just imgmatch-personal (frees its VRAM, keeps the container around for a fast `up` later)
    image-match stop                # stop both
    #>
    param(
        [Parameter(Position = 0)]
        [ValidateSet("up", "down", "stop", "restart", "logs", "build")]
        [string]$Action = "up",

        [Parameter(Position = 1)]
        [ValidateSet("general", "personal")]
        [string]$Category
    )
    Push-Location $ImageMatchDir
    try {
        switch ($Action) {
            "up" {
                if ($Category) { docker compose up -d "imgmatch-$Category" }
                else { _ImageMatchStartBoth @() }
            }
            "down"  { docker compose down }
            "stop"  {
                # Unlike "down" (which tears down the whole project -- no
                # per-category scoping, since `docker compose down` doesn't
                # take a service argument), `stop` leaves the container(s)
                # in place, just not running -- CUDA context dies with the
                # process either way, so this is the one that actually frees
                # VRAM for just one category without touching the other.
                # `image-match up [category]` brings it back with no rebuild
                # and no recreate, since nothing about the image/config changed.
                if ($Category) { docker compose stop "imgmatch-$Category" }
                else { docker compose stop }
            }
            "build" {
                # imgmatch-general and imgmatch-personal share one image
                # tag (image-match:latest) built from the identical
                # Dockerfile/context -- building both is pure duplicated
                # work (a second ~80s layer-export for the same bytes).
                # Build just one; the resulting tag covers both services.
                docker compose build imgmatch-general
            }
            "restart" {
                # `docker compose restart` deliberately does NOT pick up a
                # newly built image -- it just stops/starts the *existing*
                # container on whatever image it already had. Only `up -d`
                # notices the image changed and recreates the container.
                # --force-recreate makes that recreation unconditional
                # (don't rely on Compose's own change-detection here) --
                # this is "restart" in the sense you actually want after a
                # rebuild: give me a fresh container from the current image.
                if ($Category) { docker compose up -d --force-recreate "imgmatch-$Category" }
                else { _ImageMatchStartBoth @("--force-recreate") }
            }
            "logs" {
                if ($Category) { docker compose logs -f --tail 100 "imgmatch-$Category" }
                else { docker compose logs -f --tail 100 }
            }
        }
    } finally {
        Pop-Location
    }
}
