[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# blasdrift

将 NumPy 运算结果与独立、不依赖 BLAS 的精确参考值比较，量化浮点误差。适合检查不同 BLAS 后端、CPU 架构和线程数下的数值差异；它不会强制结果逐位一致。

## 输出内容

- 根据已报告的数值问题模式构造的测试样例。
- 使用 Python `fractions.Fraction` 计算的精确有理数参考值。
- 绝对误差、ULP 距离，以及 `exact`、`within_tolerance` 或 `drift_detected` 判定。
- 人类可读或 JSON 输出，以及可选的线程敏感性检查。

NumPy 归约和点积的结果可能随累加顺序及后端实现变化。出现差异本身不代表 BLAS 有 bug；但很小的舍入差异也可能影响后续的相等判断或 `floor()` 调用。

## 安装

需要 Python 3.9+ 和 NumPy 1.24+。从源码安装：

```bash
git clone https://github.com/zhuhroscar-tech/blasdrift.git
cd blasdrift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

`dev` extra 包含 pytest、覆盖率工具，以及用于查看后端和控制线程的 `threadpoolctl`。也可从 [Releases](https://github.com/zhuhroscar-tech/blasdrift/releases) 获取 wheel；安装前请核对随附的 `SHA256SUMS.txt`。

## 快速上手

```bash
blasdrift
blasdrift --json
blasdrift --list-fixtures
blasdrift --fixture qmcpack_near_zero_dot
blasdrift --thread-sensitivity qmcpack_near_zero_dot
blasdrift --check-drift
```

使用 `--check-drift` 时，只要选中的样例出现 `drift_detected` 就返回 `1`，否则返回 `0`。不加该参数时，普通测试结果不会导致失败退出。独立的线程敏感性模式只报告状态，不充当失败门禁；在脚本中使用时应读取输出内容。

## 范围与开发

样例数量有限，并非穷举式浮点 fuzzer。结果只反映这些输入在当前环境中的表现，不能保证你的应用可复现。请保留应用自身的数值测试；需要可复现结果时，应明确控制 BLAS 后端与线程数。blasdrift 不会修改 NumPy 或替换其算法。

安装开发依赖后运行 `python -m pytest --cov`。[样例源码](src/blasdrift/fixtures.py)列出了问题来源，[CI](.github/workflows/ci.yml)定义了 Linux/macOS 测试及构建流程。

## 许可证

[MIT](LICENSE)。
