# Pymss Studio — Python GUI

> **Python 桌面版**，替代原 Vue + Tauri 界面。直接调用 `python/worker.py` 进行音源分离，复用 `pymss` 核心推理能力。

---

## 与上游差异 (vs pymss-desktop)

本项目 fork 自 [pymss-desktop](https://github.com/pymms-studio/pymss-desktop)（原 Vue + Tauri 版本），将前端完全替换为 Python/PySide6，并**新增了 MSST 推理引擎**。

### 为什么要加 MSST 引擎？

原始 `pymss-desktop` 使用 `pymss_core`（pymss 自带分离器）进行所有模型推理。但在实际使用中发现：

1. **`mel_band_roformer` 架构 vocals 输出偏差** — `pymss_core` 的实现在人声通道存在已知的偏差，分离出的 vocal 质量明显偏低。换用 [MSST-WebUI](https://github.com/TRvlvr/application_data) （另一种音源分离 WebUI）的推理代码处理**同一模型权重**即可得到正确结果。
2. **部分架构 pymss_core 不原生支持** — 如 `segm_models`、`swin_upernet`、`bs_mamba2`、`torchseg`、`scnet_unofficial` 等架构仅在 MSST 中有实现，`pymss_core` 无法加载这些模型。
3. **模型兼容性** — `pymss_core` 对某些 ckpt 的 `state_dict` 格式限制较严格（要求 `weights_only=True`），而 MSST 引擎的加载路径（`weights_only=False`）能兼容更多社区模型。

### 解决方案

引入 **MSST 引擎适配层**（`engine/`），允许在 UI 中为同一模型一键切换引擎：

- 沿用原 MSST-WebUI 的 `get_model_from_config()` + `demix()` 推理链路
- 通过 `MsstSeparatorAdapter` 包装器暴露与 `MSSeparator.process_folder()` 兼容的接口
- 分离界面支持实时引擎切换，无需修改模型文件

---

## 快速开始

### 1. 克隆

```bash
git clone https://github.com/asp3000/pymss-studio-py.git
cd pymss-studio-py
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> 如果不需要 GPU 加速，可先安装 CPU 版 PyTorch 以减小体积：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 4. 下载模型

首次启动前需要下载分离模型。从 [ModelScope](https://modelscope.cn) 或 [HuggingFace](https://huggingface.co) 下载模型文件，放入 `data/models/` 目录下对应的子目录中。

支持通过「设置 → 模型库」页面的 **下载** 按钮自动拉取。

### 5. （可选）放置 ffmpeg / aria2c

将 `ffmpeg.exe` 和 `aria2c.exe` 放入 `bin/` 目录，worker 进程会自动从 PATH 中找到它们。  
不放置也不影响核心功能，但下载模型和输出音频编码会受限。

### 6. 启动

```bash
# 方式一：双击 start.bat（无控制台窗口）
start.bat

# 方式二：直接运行（显示控制台窗口，适合看日志）
python run.py
```

首次启动后进入「设置」页，确认：
- **Python 解释器**：指向安装了 `pymss` + torch 的 python（如果上面用 venv 创建则已自动检测）
- **worker.py 目录**：默认 `python/`，若 worker 脚本在别处请手动指定
- **数据根目录**：默认 `data/`

配置保存在根目录的 `config.json`。

---

## 系统要求

| 项目 | 最低要求 | 推荐 |
|---|---|---|
| Python | 3.10+ | 3.11 / 3.12 |
| PySide6 | 6.7+ | 6.8+ |
| GPU | — | NVIDIA CUDA (8 GB+ VRAM) |
| 内存 | 8 GB | 16 GB+ |
| 磁盘 | 10 GB 可用 | SSD, 50 GB+（用于模型 + 输出） |

---

## 功能一览

| 页面 | 功能 |
|---|---|
| **分离** | 默认启动页；支持文件/文件夹拖放；选模型、输出格式/设备/TTA，批量分离 |
| **模型库** | 列出本地模型、在线下载、删除、刷新 |
| **任务** | 进度条、实时日志、取消、清理已完成任务 |
| **结果** | 浏览输出目录、打开文件/文件夹 |
| **编辑器** | 简化版：选结果文件夹 → 波形预览 → 播放 → 导出混音 |
| **工作流** | 简化版：选工作流 YAML + 输入文件 → 运行 |
| **设置** | Python 解释器、worker 目录、pymss 路径、数据根、默认设备/格式/输出目录 |

---

## 模型运行引擎

从本版开始（基于 `pymss-desktop` 代码底子），Pymss Studio 在原始单引擎基础之上**新增 MSST 引擎**，实现**多引擎切换**，解决不同工具对同一模型的输出不一致问题。

### 引擎架构

```
分离界面 → 选择引擎（Pymss / MSST）
               │
      ┌────────┴────────┐
      ▼                 ▼
  Pymss 引擎        MSST 引擎
  (pymss_core)      (MSST-WebUI runtime)
      │                 │
      ▼                 ▼
  pymss/utils.py    MSST/utils/utils.py
  pymss_core/       MSST/modules/
```

> ⚠️ **MSST 引擎依赖**：`engine/msst/engine.py` 需要从本机 `D:\AI\MSST-WebUI` 目录导入模块（`get_model_from_config` / `demix`）。部署时需确保 MSST-WebUI 位于该路径，或修改 `MSST_ROOT` 常量指向正确位置。

### 引擎对照表

| 架构 | 可用引擎 | 说明 |
|---|---|---|
| `mel_band_roformer` | Pymss (默认), MSST | ⚠️ Pymss_core 实现在 vocals 通道有已知偏差，可切到 MSST 引擎获得正确结果 |
| `bs_roformer` | Pymss (默认), MSST | 两边均有实现 |
| `htdemucs` / `mdx23c` / `bandit` / `bandit_v2` / `scnet` / `apollo` | Pymss (默认), MSST | 两边均有实现 |
| `segm_models` / `swin_upernet` / `bs_mamba2` | **仅 MSST** | Pymss 不支持，引擎锁定不可切换 |
| `vr` / `tiger` / `bs_roformer_hyperace` / `legacy_demucs` / `legacy_tasnet` | **仅 Pymss** | MSST 不支持，引擎锁定不可切换 |

### 如何切换

1. 在「分离」页面选择模型
2. 展开「高级设置」，找到 **引擎** 下拉框
3. 选择 `Pymss`（默认引擎）或 `MSST`
4. 点击「开始分离」即可

> 当某个架构仅支持单一引擎时，引擎下拉框**禁用**，无法切换。

### 添加新引擎

在 `engine/__init__.py` 中注册：

```python
from .my_engine import MyEngine

REGISTRY["my_engine"] = MyEngine
ARCHITECTURE_ENGINES["bs_roformer"] = ["pymss", "my_engine"]
```

新引擎需实现 `load_model()` + `separate()` 接口，也可通过 `MsstSeparatorAdapter` 包装器模式将现有引擎适配为 `process_folder()` 接口（参见 `engine/__init__.py` 中的 `MsstSeparatorAdapter` 实现）。

---

## 项目结构

```
pymss-studio-py/
├── __init__.py            # 包标记
├── __main__.py            # python -m 入口
├── main.py                # Python 入口点
├── run.py                 # 启动器（含 importlib 包名映射）
├── config.py              # 配置管理（替代 Tauri store）
├── config.json            # 默认配置
├── main_window.py         # 主窗口（导航 + 视图路由）
├── worker_bridge.py       # 子进程桥接 + JSON 事件解析
├── task_model.py          # 任务数据模型
├── workflow_graph.py      # 工作流图操作
├── start.bat              # Windows 无窗口启动脚本
├── start.vbs              # VBS 无窗口启动
│
├── views/                 # 各页面视图
│   ├── separate.py        #   分离页
│   ├── models.py          #   模型库页
│   ├── workflows.py       #   工作流页
│   ├── workflow_simple.py #   简化工作流运行
│   └── settings.py        #   设置页
│
├── widgets/               # 可复用控件
│   └── drop_area.py       #   文件拖放区
│
├── engine/                # 推理引擎
│   ├── __init__.py        #   引擎注册表
│   └── msst/              #   MSST 引擎
│
├── python/                # Worker 子进程脚本
│   ├── worker.py          #   命令分发入口
│   ├── worker_infer.py    #   推理执行
│   ├── worker_models.py   #   模型管理
│   ├── worker_download.py #   模型下载
│   └── ...                #   其它 worker 模块
│
├── data/                  # 运行时数据
│   ├── models/            #   模型文件（需自行下载）
│   └── settings/          #   工作流等用户数据
│
└── bin/                   # 外部工具（可选）
    ├── ffmpeg.exe
    └── aria2c.exe
```

---

## 架构

```
PySide6 GUI (pymss_gui)
    │  调用 WorkerBridge
    ▼
WorkerBridge ── spawn ──>  python worker.py <command> --payload x.json
    │                          (原 Tauri/Rust 层做的事，这里用 Python 进程做)
    ▼  解析 stdout JSON 事件
TaskManager + 各视图更新
```

worker 每个命令起一个独立子进程，按行输出 JSON 事件；GUI 用后台线程读取并转发为 Qt 信号。  
GUI 进程与 `pymss` / `torch` 完全解耦，无需在 GUI 进程里 import 重型依赖。

---

## 从零搭建（Windows 示例）

```batch
git clone https://github.com/asp3000/pymss-studio-py.git
cd pymss-studio-py
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4
start.bat
```

首次启动后：
1. 进入「设置」→「模型库」下载所需模型
2. 或手动将 `.ckpt` / `.yaml` 文件放入 `data/models/<分类>/` 下
3. 回到「分离」页，拖入音频文件开始分离

---

## 许可

[MIT](LICENSE)
