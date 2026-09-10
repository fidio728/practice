#!/usr/bin/env bash
# deploy_06_robots_daily.sh — push fetcher + queue to 16 machines and launch.
#
# Usage:
#   bash deploy_06_robots_daily.sh push          # scp script + queue
#   bash deploy_06_robots_daily.sh launch        # ssh nohup launch on all 16
#   bash deploy_06_robots_daily.sh tail N        # tail last 40 jsonl rows from machine N
#   bash deploy_06_robots_daily.sh count         # ssh count rows in each shard's output
#   bash deploy_06_robots_daily.sh pull          # scp shard outputs back to E:
#
# Local files:
#   06_robots_daily_fetch_digests.py
#   E:/bot_paper_data/cdx_digest_to_fetch.jsonl   (~22 MB, 99,745 rows)

set -u

KEY=/c/Users/xl/Downloads/gce_key
SCRIPT_LOCAL="06_robots_daily_fetch_digests.py"
QUEUE_LOCAL="E:/bot_paper_data/cdx_digest_to_fetch.jsonl"
SCRIPT_REMOTE="06_robots_daily_fetch_digests.py"
QUEUE_REMOTE="cdx_digest_to_fetch.jsonl"

# Format: idx user@host datadir
HOSTS=(
  "0  fidio728@34.174.42.206   /home/fidio728/bot_paper_data"
  "1  fidio728@34.162.166.20   /home/fidio728/bot_paper_data"
  "2  fidio728@34.162.50.92    /home/fidio728/bot_paper_data"
  "3  fidio728@136.114.165.116 /home/fidio728/bot_paper_data"
  "4  fidio728@34.106.167.221  /home/fidio728/bot_paper_data"
  "5  fidio728@34.162.136.118  /home/fidio728/bot_paper_data"
  "6  fidio728@34.19.252.146   /home/fidio728/bot_paper_data"
  "7  fidio728@34.18.196.69    /home/fidio728/bot_paper_data"
  "8  fidio728@34.39.173.63    /home/fidio728/bot_paper_data"
  "9  fidio728@34.155.88.193   /home/fidio728/bot_paper_data"
  "10 fidio728@34.130.14.209   /home/fidio728/bot_paper_data"
  "11 root@165.245.169.237     /root/bot_paper_data"
  "12 root@144.126.210.168     /root/bot_paper_data"
  "13 root@138.68.49.186       /root/bot_paper_data"
  "14 root@192.241.244.211     /root/bot_paper_data"
  "15 root@161.35.118.247      /root/bot_paper_data"
)

NUM_SHARDS=16
WORKERS=4
RATE_LIMIT=0.7

ssh_opts="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"

push_one() {
  local idx=$1 user_host=$2 datadir=$3
  echo "[$idx] push -> $user_host:$datadir"
  ssh $ssh_opts "$user_host" "mkdir -p $datadir" || { echo "[$idx] mkdir FAILED"; return 1; }
  scp $ssh_opts "$SCRIPT_LOCAL" "$user_host:$datadir/$SCRIPT_REMOTE" || { echo "[$idx] script scp FAILED"; return 1; }
  scp $ssh_opts "$QUEUE_LOCAL"  "$user_host:$datadir/$QUEUE_REMOTE"  || { echo "[$idx] queue scp FAILED";  return 1; }
  echo "[$idx] push OK"
}

launch_one() {
  local idx=$1 user_host=$2 datadir=$3
  local logfile="$datadir/06_robots_daily_m${idx}.log"
  # -n: redirect stdin from /dev/null. Without explicit </dev/null /dev/null
  # >>logfile 2>&1 inside the remote command, ssh hangs because nohup'd
  # python3 inherits ssh's stdout/stderr fds and ssh waits for them to
  # close. Full triple redirect detaches cleanly.
  local cmd="cd $datadir && nohup python3 $SCRIPT_REMOTE \
    --shard $idx --num-shards $NUM_SHARDS \
    --data-dir $datadir \
    --workers $WORKERS --rate-limit $RATE_LIMIT \
    </dev/null >> $logfile 2>&1 & disown; echo started_pid=\$!"
  echo "[$idx] launch -> $user_host"
  ssh -n $ssh_opts "$user_host" "$cmd" \
    && echo "[$idx] launch OK (log: $logfile)" \
    || echo "[$idx] launch FAILED"
}

tail_one() {
  local idx=$1 user_host=$2 datadir=$3
  echo "=== [$idx] last 40 jsonl rows from digest_robots_cache_m${idx}.jsonl ==="
  ssh $ssh_opts "$user_host" "tail -n 40 $datadir/digest_robots_cache_m${idx}.jsonl 2>/dev/null | wc -l; echo ---; tail -n 5 $datadir/06_robots_daily_m${idx}.log 2>/dev/null"
}

count_one() {
  local idx=$1 user_host=$2 datadir=$3
  ssh $ssh_opts "$user_host" "wc -l $datadir/digest_robots_cache_m${idx}.jsonl 2>/dev/null || echo '0 (no file yet)'" | awk -v i="$idx" '{print "["i"]", $0}'
}

pull_one() {
  local idx=$1 user_host=$2 datadir=$3
  echo "[$idx] pull <- $user_host"
  scp $ssh_opts "$user_host:$datadir/digest_robots_cache_m${idx}.jsonl" "E:/bot_paper_data/digest_robots_cache_m${idx}.jsonl" || { echo "[$idx] pull FAILED"; return 1; }
  echo "[$idx] pull OK"
}

case "${1:-}" in
  push)
    for h in "${HOSTS[@]}"; do
      read -r idx user_host datadir <<< "$h"
      push_one "$idx" "$user_host" "$datadir" &
    done
    wait
    ;;
  launch)
    # Optional: bash deploy_06_robots_daily.sh launch 1 2 3 ...
    # restricts launch to only those shard indexes. Default = all.
    shift || true
    only_idxs="$*"
    for h in "${HOSTS[@]}"; do
      read -r idx user_host datadir <<< "$h"
      if [[ -n "$only_idxs" ]]; then
        match=0
        for x in $only_idxs; do
          [[ "$x" == "$idx" ]] && match=1
        done
        [[ "$match" == "1" ]] || continue
      fi
      launch_one "$idx" "$user_host" "$datadir" &
    done
    wait
    ;;
  tail)
    only_idx="${2:-}"
    for h in "${HOSTS[@]}"; do
      read -r idx user_host datadir <<< "$h"
      if [[ -z "$only_idx" || "$only_idx" == "$idx" ]]; then
        tail_one "$idx" "$user_host" "$datadir"
      fi
    done
    ;;
  count)
    for h in "${HOSTS[@]}"; do
      read -r idx user_host datadir <<< "$h"
      count_one "$idx" "$user_host" "$datadir" &
    done
    wait
    ;;
  pull)
    for h in "${HOSTS[@]}"; do
      read -r idx user_host datadir <<< "$h"
      pull_one "$idx" "$user_host" "$datadir" &
    done
    wait
    ;;
  *)
    echo "Usage: $0 {push|launch|count|tail [N]|pull}"
    exit 1
    ;;
esac
