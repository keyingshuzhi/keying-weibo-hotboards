# 微博两大榜自动化 · 热搜榜 / 社会榜（WeCom 推送 + 合并图 + AI 摘要）

> Python 3.12 · Chrome/Chromedriver 142.0.7444.60 (arm64)  
> 团队：**柯影数智团队**（西安柯影数智科技有限责任公司）  
> 说明：**公开库**，面向**内部信息汇总与学习用途**（非商业、非分发抓取服务）

---

## ✨ 功能亮点

- **两大榜（热搜/社会）原生态抓取**：不改动数据源，不做权重干预。  
- **AI 摘要（DeepSeek）**：自动生成要点与趋势点评；控制台会打印调用可用性与成功状态。  
- **Top 3 原文链接**：每个榜单固定展示前 3 条，便于快速浏览与复核。  
- **图片渲染**：优先输出**两榜合并单图**；若环境不支持自动回退为两张单图。  
- **企业微信推送（WeCom）**：自动分片发送 Markdown，支持图片/整页截图。  
- **关键词筛选**：支持「不筛选 / 科技快捷筛 / 命令行自定义 / 文件读取 / 环境变量」五种模式。  
- **安全日志**：`key/token/cookie/webhook` 自动打码；URL 仅保留 host，参数值统一 `**`。  

---

## 🧱 目录结构（沿用原项目架构）

```
weibo_hotSearch/
├─ weibo_hotsearch/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ utils.py
│  ├─ fetch.py
│  ├─ render_image.py
│  ├─ summary_deepseek.py
│  ├─ wecom.py
│  └─ screenshot.py
├─ scripts/
│  ├─ weibo_daily.py
│  ├─ headless.py
│  └─ main.py
├─ resource/
│  ├─ hot_research.jpg
│  ├─ heiti.ttf
│  └─ SmileySans.ttf
├─ info/                      # ← 放置二维码/关键词等说明资源
│  ├─ official-account.png    # 公众号：柯影效率研究站
│  ├─ service-account.png     # 服务号：柯影数智云
│  └─ fans-group.png          # 粉丝群二维码
├─ archive/                   # 输出图片（榜单合并图/单图）
├─ screenshot/                # 网页整页截图
├─ requirements.txt
└─ .env.example
```

---

## 🚀 环境与安装

**要求：**
- Python **3.12**
- Google Chrome **142.0.7444.60 (arm64)** 与 **chromedriver** 版本一致
- macOS/arm64 验证通过（其他平台请自行评估）

```bash
git clone <your-repo-url> weibo_hotSearch
cd weibo_hotSearch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # 按需填写 DeepSeek、WeCom、Chromedriver 等配置
```

---

## ⚙️ 配置说明（.env）

`.env.example` 中包含常见配置项，典型字段如下（按需启用）：

```env
# DeepSeek
DEEPSEEK_API_KEY=sk-****************
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TIMEOUT_SEC=90
DEEPSEEK_RETRIES=2

# 企业微信机器人（传 key 或完整 webhook url 二选一）
WECOM_WEBHOOK_KEY_OR_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=****************

# ====== 微博 Cookie（两种方式，优先读取文件）======
WEIBO_COOKIE=

# ====== 时区 ======
TZ=Asia/Shanghai

# 浏览器与截图
CHROMEDRIVER=/absolute/path/to/chromedriver
SCREENSHOT_TIMEOUT=20
NO_HEADLESS=false
NO_COOKIE_INJECT=false

# 输出目录
OUTDIR=archive
SCREENSHOT_DIR=screenshot

# 关键词控制（可留空；也可用命令行/文件/环境变量覆盖）
DEFAULT_TOPN=10
DEFAULT_TECH_ONLY=true
AUTO_SUMMARY=true
AUTO_WECHAT=true
MULTI_BOARDS=true
```

> 运行时，**敏感字段会自动打码**打印到控制台（URL 仅保留 host）。

---

## 🧭 典型用法（五种关键词策略）

> 下列命令默认从仓库根目录执行：`python scripts/main.py ...`

### 1) 原生态（不筛选）
```bash
python scripts/main.py boards --topn 10 --summary --wecom
```

### 2) 科技领域快捷筛选（使用 `CONFIG.tech_keywords` 预置词表）
```bash
python scripts/main.py boards --topn 10 --summary --wecom --tech-only
```

### 3) 自定义关键词（命令行直写）
```bash
python scripts/main.py boards --keywords "AI, 芯片, 半导体, 英伟达, 华为" --topn 10 --summary --wecom
```

### 4) 从文件读取关键词（每行一个或逗号分隔）
```bash
python scripts/main.py boards --keywords-file ./keywords.txt --topn 10 --summary
```

### 5) 用环境变量（适合 cron）
```bash
KEYWORDS="AI,芯片,大模型" python scripts/main.py boards --topn 10 --summary --wecom
```

> **DeepSeek 状态打印样例**（控制台）  
> `[DeepSeek] available=True called=True success=True model=deepseek-chat base=https://api.deepseek.com/**`

---

## 🖼️ 输出内容

- **Markdown**：AI 摘要（限长） + 两榜 Top 3 超链接（各 3 条，共 6 条）  
- **图片**：优先两榜**合并单图**，若无法合并则回退为两张单图  
- **截图（可选）**：微博热搜页整页截图  

默认输出目录：
```
archive/      # 图片 → 例：20YY年MM月DD日HH:MM.两大榜.png
screenshot/   # 整页截图
```

---

## ⏰ 定时任务（crontab 示例）

每天 08:00 推送，关键词从环境变量读取：

```bash
# 编辑 cron
crontab -e

# 每天 08:00 运行（注意替换你的虚拟环境与仓库路径）
0 8 * * * cd /path/to/weibo_hotSearch &&   /path/to/.venv/bin/python scripts/main.py boards   --topn 15 --summary --wecom   >> /path/to/weibo_hotSearch/cron.log 2>&1
```

> 如需科技类筛选：在命令中追加 `--tech-only`；或使用 `KEYWORDS="..."` 包裹在同一行命令前缀。

---

## 🧩 进阶与约定

- **渲染合并图**：若 `render_two_boards_image` 可用则直接合成；否则先各自出图再由 Pillow 竖向拼接，并在页眉绘制时间戳与标签。  
- **关键词匹配**：为简单包含匹配，建议保持词表合理规模。  
- **控制台打码**：`key/token/cookie/webhook` 等统一打 `**`；URL 仅保留 host，参数值改写为 `**`。  
- **字体与美观**：`resource/heiti.ttf` 与 `resource/SmileySans.ttf` 需可读，确保中文/数字排版效果稳定。  

---

## ❓ 常见问题（FAQ）

- **为什么合并图生成失败？**  
  缺少 Pillow 或字体/资源路径错误会回退为两张单图；请检查 `resource/` 及日志提示。  

- **DeepSeek 摘要失败？**  
  检查 `.env`、网络、`DEEPSEEK_API_KEY` 与 `DEEPSEEK_BASE_URL`。控制台会打印 `available/called/success` 与掩码后的 base。  

- **WeCom 推送失败？**  
  核对 webhook、网络连通与消息长度；本工具会自动分片超长 Markdown。  

---

## 🧑‍🤝‍🧑 团队与社群

**柯影数智团队**  
**西安柯影数智科技有限责任公司**

<table>
  <tr>
    <td align="center">
      <b>微信公众号｜柯影效率研究站</b><br/>
      <img src="info/official-account.png" alt="柯影效率研究站 - 公众号二维码" width="220"/>
    </td>
    <td align="center">
      <b>微信服务号｜柯影数智云</b><br/>
      <img src="info/service-account.png" alt="柯影数智云 - 服务号二维码" width="220"/>
    </td>
    <td align="center">
      <b>粉丝群｜加入社群交流</b><br/>
      <img src="info/fans-group.png" alt="粉丝群二维码" width="220"/>
    </td>
  </tr>
</table>

> 欢迎关注公众号/服务号、加入粉丝群，获取更多自动化、数据智能与 AIGC 实践分享。

---

## 🔒 合规与声明

- 抓取内容版权归原平台与作者所有；本项目仅用于**学习研究与内部信息汇总**。  
- 使用时请遵循微博、企业微信、DeepSeek 等平台条款与开发者协议；禁止用于商业化抓取或对外分发服务。  
- 项目默认不持久化任何账号凭据；如需注入 Cookie，请确保来源合法与内部合规。  

---

## 📜 License

本项目以 **MIT License** 开源发布（见仓库 `LICENSE` 文件）。  
**定位：公开库，但仅建议用于内部信息汇总与学习用途。**

---

## 🙏 致谢

- 微博热搜公开页面  
- DeepSeek 大模型接口  
- Pillow / Requests 等优秀开源组件

---
