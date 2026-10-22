# Git fetch 代理排查（2026-10-07）

## 现象与原因

用户运行 `git fetch origin`，连接 github.com:443 超时。当前 origin 是 HTTPS；Git 配置没有显式代理。本次检查发现 AtlasCore_amd64 监听 6696，旧约定中的 7890、7897、10808、10809 均未监听。不能据此确认用户终端的全部环境，但已验证显式使用当前端口可以连通。

## 已修复与验证

使用 Windows PowerShell、Git，临时指定代理，不修改全局配置：

```powershell
git -c http.proxy=http://127.0.0.1:6696 -c http.sslBackend=openssl fetch origin
```

先用相同参数运行 `ls-remote origin refs/heads/main` 成功；fetch 首次因沙箱不能写 `.git/FETCH_HEAD` 失败，经授权后成功，origin/main 从 60dcb0a 更新为 ddf118e3b19b0779f1535b93bf346d5680c3d7d6。连接与拉取阻塞已解决。端口是本次机器实测值，代理配置变化后需重新确认。

## 待确认与限制

当前分支 xiaoji/imgo2_real，HEAD f1742db；fetch 后 `git rev-list --left-right --count origin/main...HEAD` 为 10 / 1。fetch 当时未执行 rebase；随后已保留全部模型改动并完成 rebase，见 [完成记录](rebase_main_2026-10-07.md)。未执行 push 或训练。本记录已从 stash 恢复，stash 仍保留作备份，无需再 pop。
