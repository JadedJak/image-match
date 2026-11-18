# image-match convenience commands for the Fedora/Podman laptop -- bash
# equivalent of ../powershell-aliases.ps1, same commands and arguments.
#
# Install: add this one line to ~/.bashrc, then open a new terminal:
#     source ~/image-match/laptop/image-match.sh
#
#   image-match [up|down|stop|restart|logs|build] [general|personal]
#   indexer general|personal|all
#   matchfind general|personal
#
# As on Windows: any app/ change needs `image-match build` then
# `image-match restart`; restart alone keeps running the old code.

IMAGE_MATCH_LAPTOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_imgmatch_compose() {
    (cd "$IMAGE_MATCH_LAPTOP_DIR" && podman compose "$@")
}

_imgmatch_service() {
    case $1 in
        "") ;;
        general|personal) echo "imgmatch-$1" ;;
        *) echo "image-match: unknown category '$1' (general|personal)" >&2; return 1 ;;
    esac
}

# Both UIs started together load the CLIP model at the same moment (RAM
# shared with the GPU and with Ollama) -- start general, wait until it's
# listening, then personal. Same as the PowerShell version.
_imgmatch_wait_ready() {
    local svc=$1 i
    for ((i = 0; i < 100; i++)); do
        _imgmatch_compose logs "$svc" 2>&1 | grep -qE 'NiceGUI ready|Running on local URL' && return 0
        [[ -n $(podman ps -q --filter "label=com.docker.compose.service=$svc" --filter status=running) ]] || return 1
        sleep 3
    done
    return 1
}

_imgmatch_start_both() {
    local svc
    for svc in imgmatch-general imgmatch-personal; do
        echo "== starting $svc"
        _imgmatch_compose up -d "$@" "$svc" || return 1
        _imgmatch_wait_ready "$svc" || { echo "!! $svc did not come up -- check: image-match logs ${svc#imgmatch-}" >&2; return 1; }
        echo "== $svc ready"
    done
}

image-match() {
    local action=${1:-up} svc
    svc=$(_imgmatch_service "$2") || return 1
    case $action in
        up)      if [[ -n $svc ]]; then _imgmatch_compose up -d $svc; else _imgmatch_start_both; fi ;;
        down)    _imgmatch_compose down ;;
        stop)    _imgmatch_compose stop $svc ;;
        # one image tag serves both services -- build it once
        build)   _imgmatch_compose build imgmatch-general ;;
        # recreate, not restart: only that picks up a rebuilt image
        restart) if [[ -n $svc ]]; then _imgmatch_compose up -d --force-recreate $svc; else _imgmatch_start_both --force-recreate; fi ;;
        logs)    _imgmatch_compose logs -f --tail 100 $svc ;;
        *) echo "image-match: unknown action '$action' (up|down|stop|restart|logs|build)" >&2; return 1 ;;
    esac
}

indexer() {
    case $1 in
        general|personal) _imgmatch_compose run --rm "imgmatch-$1" python -m app.indexer ;;
        # sequential, never concurrent -- see README
        all) indexer general && indexer personal ;;
        *) echo "indexer: category required (general|personal|all)" >&2; return 1 ;;
    esac
}

matchfind() {
    local svc
    svc=$(_imgmatch_service "$1") || return 1
    [[ -n $svc ]] || { echo "matchfind: category required (general|personal)" >&2; return 1; }
    _imgmatch_compose run --rm "$svc" python -m app.bulk_match
}
