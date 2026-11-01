# 监控 5 分片 06_5.py 集群（AWS + 3 GCP）

> **给监控 Claude 的指令**：完整阅读这个文件，理解你要监控什么、报警条件、如何执行。**只做监控和报警，不要改代码、不要 kill 任何进程**——除非用户在对话里明确要求。

---

## 集群概况

5 台机器并行跑 `06_5.py`，每台抓 122k 任务的 1/5（按 stable hash 分片）。每台输出独立文件 `sitemap_url_paths_v2_m{ID}.jsonl`，到 `/home/ec2-user/data/`。

| Shard | 机器 | Region/Zone | 公网 IP | SSH 方法 | 状态文件 |
|---|---|---|---|---|---|
| **m0** | AWS EC2 | eu-north-1 | 51.20.134.228 | pem | `/home/ec2-user/data/sitemap_url_paths_v2_m0.jsonl` |
| **m1** | GCP `instance-20260427-124030` | us-central1-f | 34.59.108.199 | gcloud / pem | `/home/ec2-user/data/sitemap_url_paths_v2_m1.jsonl` |
| **m2** | GCP `instance-20260427-125728` | us-west1-b | 136.117.177.231 | gcloud / pem | `/home/ec2-user/data/sitemap_url_paths_v2_m2.jsonl` |
| **m3** | GCP `instance-20260427-125233` | us-east1-c | 34.24.165.222 | gcloud / pem | `/home/ec2-user/data/sitemap_url_paths_v2_m3.jsonl` |
| **m4** | GCP `instance-20260427-222900` | us-south1-c | 8.230.105.129 | gcloud / pem | `/home/ec2-user/data/sitemap_url_paths_v2_m4.jsonl` |

启动时间：约 2026-04-27 13:30 UTC。脚本配置：3w/4.0s，预期单机 2.75 clean/min。

---

## 一次性 setup：取得 GCP SSH 密钥

**这一步只在你（监控 Claude）第一次跑时做一次。**

GCP 的 SSH 密钥在用户的 GCP Cloud Shell 里 `~/.ssh/google_compute_engine`。要从本地 Windows 访问 GCP 实例，需要先把那两个文件拉下来。

**两个方案，让用户挑**：

### 方案 A：让用户从 Cloud Shell 下载密钥

让用户：
1. 打开 GCP Cloud Shell（https://console.cloud.google.com 右上角 `>_` 图标）
2. 在 Cloud Shell 跑：`cat ~/.ssh/google_compute_engine`，复制整段输出
3. 在 Cloud Shell 跑：`cat ~/.ssh/google_compute_engine.pub`，复制整段输出
4. 你把第 2 段（私钥）写到 `C:\Users\xl\Downloads\gce_key`，第 3 段（公钥）写到 `C:\Users\xl\Downloads\gce_key.pub`
5. 在 Git Bash 跑 `chmod 600 /c/Users/xl/Downloads/gce_key`

之后用：
```bash
ssh -i /c/Users/xl/Downloads/gce_key fidio728@<公网IP>
```

### 方案 B：让用户安装本地 gcloud CLI

让用户去 https://cloud.google.com/sdk/docs/install-sdk 装 Windows 版（不需要管理员），跑 `gcloud auth login` 一次，然后直接：

```bash
gcloud compute ssh instance-20260427-124030 --zone us-central1-f --command="..."
```

**优先推荐方案 A**（更快，无需安装）。开始监控之前先问用户用哪个方案，按方案 A 默认。

---

## 监控 SSH 命令模板

### AWS (m0) —— 你已经有 pem
```bash
KEY=/c/Users/xl/Downloads/fidio.pem
HOST=ec2-user@51.20.134.228
ssh -o StrictHostKeyChecking=no -i "$KEY" "$HOST" "<command>"
```

### GCP m1/m2/m3 —— 用方案 A 拉下来的 GCE 密钥
```bash
GCE_KEY=/c/Users/xl/Downloads/gce_key
GCP1=fidio728@34.59.108.199    # m1, us-central1-f
GCP2=fidio728@136.117.177.231  # m2, us-west1-b
GCP3=fidio728@34.24.165.222    # m3, us-east1-c
GCP4=fidio728@8.230.105.129    # m4, us-south1-c
ssh -o StrictHostKeyChecking=no -i "$GCE_KEY" "$GCP1" "<command>"
```

注：用户 GCP 账号是 `fidio728`，不是邮箱前缀其他的。

---

## 标准监控命令

每次巡检对每台机器跑这 3 个：

```bash
# 1. 进程是否活
ps -C python3 -o pid,etime,pcpu,pmem,rss,cmd | grep 06_5 || echo NO_PROC

# 2. 日志最后几行
tail -n 8 /home/ec2-user/06_5.log 2>/dev/null || tail -n 8 ~/06_5.log

# 3. 输出文件大小 + 行数
ls -lh /home/ec2-user/data/sitemap_url_paths_v2_m<ID>.jsonl 2>/dev/null
wc -l /home/ec2-user/data/sitemap_url_paths_v2_m<ID>.jsonl 2>/dev/null
```

把 `<ID>` 替换成 0/1/2/3。

---

## 一次扫 4 台的 batch 脚本

每次巡检建议跑这个（确保 KEY/GCE_KEY 变量已设）：

```bash
KEY=/c/Users/xl/Downloads/fidio.pem
GCE_KEY=/c/Users/xl/Downloads/gce_key

declare -A MACHINES=(
  ["m0"]="-i $KEY ec2-user@51.20.134.228"
  ["m1"]="-i $GCE_KEY fidio728@34.59.108.199"
  ["m2"]="-i $GCE_KEY fidio728@136.117.177.231"
  ["m3"]="-i $GCE_KEY fidio728@34.24.165.222"
  ["m4"]="-i $GCE_KEY fidio728@8.230.105.129"
)

for m in m0 m1 m2 m3 m4; do
  ID=${m:1}
  echo "=== $m ==="
  ssh -o StrictHostKeyChecking=no ${MACHINES[$m]} "
    ps -C python3 -o pid,etime,pcpu,pmem,rss | grep -v PID | head -1 || echo 'NO_PROC'
    echo '---'
    tail -n 5 ~/06_5.log 2>/dev/null
    echo '---'
    wc -l /home/ec2-user/data/sitemap_url_paths_v2_m${ID}.jsonl 2>/dev/null || echo '0 lines'
  "
done
```

---

## 报警条件

任何一台触发以下任一条件，**立即推送通知**给用户（用 PushNotification 工具）：

### 致命（立即推 + 文字详情建议）

1. **进程消失** — `ps` 输出 `NO_PROC` 且不是因为 `done.` 自然结束
   - 推送内容：`m{ID} dead. Last log: <最后 3 行>. PID was <过去PID>.`
   - 用户可能要重启那台
2. **日志出现 `Traceback`** — 异常崩溃
   - 推送内容：`m{ID} Traceback: <截取关键行>`
3. **15+ 分钟无新行** — 持久化记录每台 `last_lines + last_change_ts` 到 `c:\Users\xl\AppData\Local\Temp\06_5_monitor_state_v3.json`，对比超 900s 报警

### 关注（有则提，但不一定推送）

1. **clean 率断崖式下降** —— 如果某次抽样 clean 率连续 30 分钟低于 30%，archive.org 大概率把那个 IP 节流了。可建议用户停那台等 IP 冷却或继续观察
2. **磁盘满** —— `df -h /home/ec2-user/data` 看到 `Use% > 90%`
3. **某台 OOM** —— `dmesg | grep -i "out of memory" | tail -3` 看到最近 30 分钟有 OOM kill 记录

---

## 巡检节奏

- **首 2 小时**：每 10 分钟查一次（IP 冷启动期最容易出问题）
- **稳定后**：每 30 分钟一次
- **跑了 24h 后**：每 1–2 小时一次

不要每分钟轮询——你这个监控会话占着用户的对话窗口，太频繁就没意义。

---

## 进度报告格式（每次巡检的输出）

简洁一行表，类似：

```
[14:30Z] m0: PID 27610 etime 1d2h, lines 42301 (+812 since last), shard 0/5 ✓
        m1: PID 3993  etime 1d1h, lines 38104 (+754), shard 1/5 ✓
        m2: PID 4012  etime 1d1h, lines 37842 (+701), shard 2/5 ✓
        m3: PID 4030  etime 1d1h, lines 39201 (+822), shard 3/5 ✓
        合计 final clean est: ~120k / 122k = 98%, ETA ~6h
```

正常状态只输出这一行总结即可，不要重复粘贴几百行 ssh 输出。

---

## 完工判定（每台独立）

某台日志最后看到：
```
Done. N processed. clean=X, empty=Y, dirty=Z
Re-run this script to retry remaining 'retryable' triples.
Stop when 'retryable' queue stops shrinking, then run Step 7.
```

→ 表示这台**第一遍**跑完了。报告给用户，问用户要不要：
- 立即用同样命令再跑一次（重试 retryable 队列）
- 还是等其他机器也跑完一遍后统一处理

**不要自动重启**——让用户决定，因为再跑一遍要烧更多 GCP credit。

---

## 当所有 4 台都至少跑完一遍

通知用户："4 台 m0/m1/m2/m3 都已完成首轮。建议合并各台输出 + 决定要不要再跑一轮 retry。" 并提示用户：

- 数据合并步骤在 `iridescent-crunching-manatee.md` 计划文件 Step 5
- 跑完的 GCP 实例**记得 stop**：
  ```
  gcloud compute instances stop instance-20260427-124030 --zone us-central1-f
  gcloud compute instances stop instance-20260427-125728 --zone us-west1-b
  gcloud compute instances stop instance-20260427-125233 --zone us-east1-c
  gcloud compute instances stop instance-20260427-222900 --zone us-south1-c
  ```
  （或在 GCP Console 手动停止）

---

## 工作交接说明

如果用户切换会话/电脑/IP，新 Claude 实例可以从这个文件无缝接手——只要这 4 个 IP 没变 + 私钥还在原位就行。状态文件 `06_5_monitor_state_v3.json` 也会自动加载历史。

如果有 IP 变更（比如 GCP 实例重启拿了新 IP），用户会告诉你新 IP，你更新这个文件里的 IP 表即可。

---

## 不在你职责范围

- 改 `06_5.py` 代码
- 调整 worker 数 / RATE_LIMIT
- kill 任何进程
- 重启实例
- 创建新实例
- 删除/合并输出文件

这些**都需要用户明确指示**才做。
