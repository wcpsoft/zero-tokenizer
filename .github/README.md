# CI/CD 流程

本项目使用GitHub Actions进行持续集成和持续部署(CI/CD)。

## 工作流程

### CI 工作流 (`.github/workflows/ci.yml`)

当代码推送到`main`或`master`分支，或创建针对这些分支的Pull Request时，会触发CI工作流。CI工作流包括以下步骤：

1. **多版本Rust测试**：在stable、beta和nightly版本的Rust上运行测试
2. **代码质量检查**：
   - 使用`cargo fmt`检查代码格式
   - 使用`cargo clippy`进行静态代码分析
3. **单元测试**：运行所有Rust单元测试
4. **Python绑定测试**：
   - 构建Python包
   - 运行Python测试套件
5. **多平台构建**：在Ubuntu、Windows和macOS上构建Python包
6. **性能基准测试**：运行BBPE分词器性能测试并保存结果

### 发布工作流 (`.github/workflows/publish.yml`)

当创建新的版本标签(如`v1.0.0`)时，会触发发布工作流。发布工作流包括以下步骤：

1. **完整测试**：运行与CI工作流相同的测试套件
2. **多平台构建**：在Ubuntu、Windows和macOS上构建Python包
3. **发布到PyPI**：将构建的包发布到Python包索引(PyPI)

## 本地开发

### 运行测试

在本地运行所有测试：

```bash
# 运行Rust测试
cargo test

# 构建Python包
maturin develop

# 运行Python测试
cd tests/python
python -m pytest -v
```

### 代码质量检查

```bash
# 检查代码格式
cargo fmt -- --check

# 运行静态分析
cargo clippy -- -D warnings
```

### 运行性能基准测试

```bash
# 构建Python包
maturin develop

# 运行BBPE性能测试
python .benchmarks/bbpe_performance.py

# 运行BBPE与GPT-2对比测试（需要安装transformers库）
pip install transformers
python .benchmarks/bbpe_vs_gpt2.py
```

## 发布流程

1. 更新版本号：
   - 在`Cargo.toml`中更新版本号
   - 在`python/zero_tokenizer/__init__.py`中更新版本号（如果存在）

2. 创建并推送标签：
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

3. GitHub Actions将自动：
   - 运行所有测试
   - 构建多平台Python包
   - 发布到PyPI（需要配置`PYPI_API_TOKEN`密钥）

## 环境变量和密钥

### PyPI API Token

为了能够发布到PyPI，需要在GitHub仓库的Settings > Secrets中配置`PYPI_API_TOKEN`密钥：

1. 在PyPI上创建API令牌
2. 在GitHub仓库设置中添加名为`PYPI_API_TOKEN`的密钥
3. 将PyPI API令牌值作为密钥值

## 性能基准测试结果

性能基准测试结果会保存在GitHub Actions的artifacts中，可以在Actions页面的相应运行中下载。

## 故障排除

### 常见问题

1. **构建失败**：检查`Cargo.toml`中的依赖是否兼容
2. **测试失败**：查看测试日志确定失败原因
3. **发布失败**：确认PyPI API令牌是否有效且权限正确

### 调试技巧

1. 在GitHub Actions中启用调试日志：
   ```yaml
   - name: Debug step
     run: |
       echo "Debug info"
       env
   ```

2. 在本地复现CI环境：
   ```bash
   # 使用Docker运行与CI相同的环境
   docker run -it -v $(pwd):/workspace rust:latest bash
   ```

## 贡献指南

1. Fork项目
2. 创建功能分支：`git checkout -b feature/new-feature`
3. 提交更改：`git commit -am 'Add new feature'`
4. 推送分支：`git push origin feature/new-feature`
5. 创建Pull Request

CI工作流将自动运行测试和代码质量检查。确保所有检查通过后再请求合并。