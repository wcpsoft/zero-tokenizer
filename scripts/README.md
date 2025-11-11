# 开发脚本

本目录包含了zero-tokenizer项目的开发脚本，用于简化环境设置、编译和安装过程。

## 脚本说明

### 1. init_uv.sh
初始化uv环境，创建Python虚拟环境并安装基本依赖。

```bash
./scripts/init_uv.sh
```

功能：
- 检查并安装uv（如果未安装）
- 创建Python虚拟环境
- 安装maturin和pytest等基本依赖

### 2. build.sh
编译Rust代码并构建Python扩展。

```bash
./scripts/build.sh
```

功能：
- 编译Rust库（release模式）
- 使用maturin构建Python扩展

### 3. install.sh
安装Python包到虚拟环境。

```bash
./scripts/install.sh
```

功能：
- 使用uv或pip以开发模式安装Python包

### 4. dev_setup.sh
完整的开发环境设置脚本，按顺序执行所有必要步骤。

```bash
./scripts/dev_setup.sh
```

功能：
- 初始化uv环境
- 激活虚拟环境
- 编译Rust代码
- 安装Python包
- 运行测试

## 使用建议

1. 首次设置环境时，直接运行：
   ```bash
   ./scripts/dev_setup.sh
   ```

2. 日常开发中，如果修改了Rust代码，运行：
   ```bash
   ./scripts/build.sh
   ```

3. 如果需要重新安装Python包，运行：
   ```bash
   ./scripts/install.sh
   ```

## 注意事项

- 所有脚本都应在项目根目录下运行
- 脚本会自动检测并使用现有的虚拟环境
- 如果uv不可用，脚本会回退到使用pip