# Windows环境下UV Python 项目管理完整流程指南

## 1. 安装 uv

```bash
# 通过 pip 安装 uv（没安装过uv的才需要这一步）。
pip install uv
# 安装后验证。
uv --version
# 输出示例：uv 0.10.10
```

------

## 2. 初始化项目

```bash
# 初始化一个名为 `weather`的新 Python 项目。`uv init`会创建一个新的目录 `weather`，并在其中生成基本的项目文件（如 `pyproject.toml`），用于管理项目的依赖和元数据。
uv init weather
```

------

## 3. 进入项目目录

```bash
# 进入刚刚创建的 `weather`项目目录，后续的命令都会在这个目录下执行。
cd weather
```

------

## 4. 创建虚拟环境

```bash
# 在当前目录中创建一个 Python 虚拟环境。执行后，会使用系统中找到的 Python 解释器，在 `.venv`文件夹中建立独立的虚拟环境。
uv venv
```

------

## 5. 激活虚拟环境

```bash
# 激活虚拟环境。运行后，当前终端会话会切换到该虚拟环境，后续安装的包和运行的 Python 都会局限在这个环境中，避免与全局 Python 环境冲突。激活后，命令行提示符变为 `(weather) → weather git:(master) x`，其中的 `(weather)`表示虚拟环境已生效。
.venv\Scripts\activate
```

------

## 6. 同步项目依赖（团队协作场景）

```bash
# 克隆项目后，可执行此命令一键还原环境。
uv sync
```

------

## 7. 添加新依赖

```bash
# 添加依赖，此时 `pyproject.toml`被更新，`uv.lock`也会更新。
uv add 你要添加的依赖
# 然后运行 sync 来应用更改。
uv sync
```

------

## 8. 避坑指南：操作对比示例

假设你的项目目录是 `my_project`，其中已有 `.venv`虚拟环境。

| 你在终端中执行的操作                                         | `uv sync`安装到哪里？  | `uv pip install requests`安装到哪里？ |
| ------------------------------------------------------------ | ---------------------- | ------------------------------------- |
| 在项目目录下，**未激活**虚拟环境                             | **项目内的 `.venv`** ✅ | **全局Python** ❌ (污染了全局环境)     |
| 在项目目录下，**已激活**虚拟环境 (`source .venv/bin/activate`或 `.venv\Scripts\activate`) | **项目内的 `.venv`** ✅ | **项目内的 `.venv`** ✅                |
| 在非项目目录下                                               | 报错（找不到项目）     | **全局Python** ❌                      |

**重要提示**：

- 

  `uv`速度更快。

- 

  尽量使用 `uv add`和 `uv sync`。

- 

  避免使用 `uv pip install`。