#!/usr/bin/env python3
"""
Zero Tokenizer vs HuggingFace Tokenizers 完整性能对比基准测试

测试所有算法：BPE、BBPE、Unigram、WordPiece
包含特性：字典初始化、批量编码、训练、解码

运行方式：
    python benchmarks/compare_with_hf.py
    python benchmarks/compare_with_hf.py --algorithm all --vocab-size 5000
    python benchmarks/compare_with_hf.py --algorithm bpe --iterations 10
"""

import time
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

# 导入Zero Tokenizer
try:
    from zero_tokenizer import (
        BPETokenizer as ZeroBPE,
        BBPETokenizer as ZeroBBPE,
        UnigramTokenizer as ZeroUnigram,
        WordPieceTokenizer as ZeroWordPiece,
    )
    ZERO_AVAILABLE = True
except ImportError as e:
    print(f"❌ Zero Tokenizer未安装: {e}")
    print("   请先运行: maturin develop")
    ZERO_AVAILABLE = False

# 导入HuggingFace Tokenizers
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    HF_AVAILABLE = True
except ImportError:
    print("⚠️  HuggingFace tokenizers未安装，将跳过HF测试")
    print("   安装命令: pip install tokenizers")
    HF_AVAILABLE = False


class ComprehensiveBenchmark:
    """完整的性能基准测试类"""

    # 算法映射
    ALGORITHMS = {
        'bpe': 'BPE',
        'bbpe': 'BBPE (Byte-level BPE)',
        'unigram': 'Unigram',
        'wordpiece': 'WordPiece',
    }

    def __init__(self, algorithm: str = 'bpe', vocab_size: int = 1000, iterations: int = 5):
        self.algorithm = algorithm
        self.vocab_size = vocab_size
        self.iterations = iterations
        self.results = {}

        # 测试数据集
        self.train_corpus = self._generate_corpus(500)
        self.test_texts = self._generate_corpus(100)

        # 字典路径
        self.dict_path = project_root / "dict" / "常用汉字字表.txt"

    def _generate_corpus(self, size: int) -> List[str]:
        """生成测试语料（包含英文和中文）"""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a high-level programming language.",
            "Machine learning models require large datasets.",
            "Natural language processing is fascinating.",
            "Tokenization is the first step in NLP pipelines.",
            "Performance optimization is crucial for production systems.",
            "Rust provides memory safety without garbage collection.",
            "Zero Tokenizer aims to match HuggingFace performance.",
            "人工智能正在改变世界。",
            "深度学习需要大量的计算资源。",
            "自然语言处理是计算机科学的重要分支。",
            "分词是文本处理的第一步。",
        ]

        corpus = []
        for i in range(size):
            corpus.append(texts[i % len(texts)])
        return corpus

    def _create_zero_tokenizer(self):
        """创建Zero Tokenizer实例"""
        if not ZERO_AVAILABLE:
            return None

        if self.algorithm == 'bpe':
            return ZeroBPE()
        elif self.algorithm == 'bbpe':
            return ZeroBBPE()
        elif self.algorithm == 'unigram':
            return ZeroUnigram()
        elif self.algorithm == 'wordpiece':
            return ZeroWordPiece()
        else:
            raise ValueError(f"未知算法: {self.algorithm}")

    def _create_hf_tokenizer(self):
        """创建HuggingFace Tokenizer实例"""
        if not HF_AVAILABLE:
            return None, None

        if self.algorithm == 'bpe':
            tokenizer = Tokenizer(models.BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["[UNK]"]
            )
        elif self.algorithm == 'bbpe':
            tokenizer = Tokenizer(models.BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["[UNK]"]
            )
        elif self.algorithm == 'unigram':
            tokenizer = Tokenizer(models.Unigram())
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.UnigramTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["[UNK]"]
            )
        elif self.algorithm == 'wordpiece':
            tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.WordPieceTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["[UNK]"]
            )
        else:
            raise ValueError(f"未知算法: {self.algorithm}")

        return tokenizer, trainer

    def benchmark_training(self) -> tuple:
        """基准测试：训练速度"""
        print("\n" + "="*70)
        print(f"📊 训练速度基准测试 - {self.ALGORITHMS[self.algorithm]}")
        print("="*70)

        tokenizer_zero = None
        tokenizer_hf = None

        # Zero Tokenizer 训练
        if ZERO_AVAILABLE:
            times = []
            for i in range(self.iterations):
                tokenizer = self._create_zero_tokenizer()

                start = time.perf_counter()
                tokenizer.train(self.train_corpus, self.vocab_size)
                elapsed = time.perf_counter() - start

                times.append(elapsed)
                print(f"  Zero Tokenizer 第{i+1}次: {elapsed*1000:.2f}ms")

            avg_time = sum(times) / len(times)
            self.results['zero_training'] = {
                'avg_ms': avg_time * 1000,
                'min_ms': min(times) * 1000,
                'max_ms': max(times) * 1000,
            }
            print(f"  ✅ Zero Tokenizer 平均: {avg_time*1000:.2f}ms")

            # 保留训练好的tokenizer
            tokenizer_zero = self._create_zero_tokenizer()
            tokenizer_zero.train(self.train_corpus, self.vocab_size)

        # HuggingFace Tokenizer 训练
        if HF_AVAILABLE:
            times = []
            for i in range(self.iterations):
                tokenizer, trainer = self._create_hf_tokenizer()

                start = time.perf_counter()
                tokenizer.train_from_iterator(self.train_corpus, trainer=trainer)
                elapsed = time.perf_counter() - start

                times.append(elapsed)
                print(f"  HF Tokenizer 第{i+1}次: {elapsed*1000:.2f}ms")

            avg_time = sum(times) / len(times)
            self.results['hf_training'] = {
                'avg_ms': avg_time * 1000,
                'min_ms': min(times) * 1000,
                'max_ms': max(times) * 1000,
            }
            print(f"  ✅ HF Tokenizer 平均: {avg_time*1000:.2f}ms")

            # 保留训练好的tokenizer
            tokenizer_hf, trainer = self._create_hf_tokenizer()
            tokenizer_hf.train_from_iterator(self.train_corpus, trainer=trainer)

        return tokenizer_zero, tokenizer_hf

    def benchmark_dict_init(self) -> Optional[float]:
        """基准测试：字典初始化（仅Unigram和WordPiece）"""
        if self.algorithm not in ['unigram', 'wordpiece']:
            return None

        if not self.dict_path.exists():
            print(f"  ⚠️  字典文件不存在: {self.dict_path}")
            return None

        print("\n" + "="*70)
        print(f"📊 字典初始化速度测试 - {self.ALGORITHMS[self.algorithm]}")
        print("="*70)

        if ZERO_AVAILABLE:
            times = []
            for i in range(self.iterations):
                start = time.perf_counter()
                tokenizer = self._create_zero_tokenizer()
                elapsed = time.perf_counter() - start

                times.append(elapsed)
                print(f"  Zero Tokenizer 第{i+1}次: {elapsed*1000:.2f}ms")

            avg_time = sum(times) / len(times)
            self.results['zero_dict_init'] = {
                'avg_ms': avg_time * 1000,
                'min_ms': min(times) * 1000,
                'max_ms': max(times) * 1000,
            }
            print(f"  ✅ Zero Tokenizer 平均: {avg_time*1000:.2f}ms")
            return avg_time

        return None

    def benchmark_encoding_single(self, tokenizer_zero=None, tokenizer_hf=None) -> None:
        """基准测试：单条编码速度"""
        print("\n" + "="*70)
        print(f"📊 单条编码速度测试 - {self.ALGORITHMS[self.algorithm]}")
        print("="*70)

        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "人工智能正在改变世界。",
            "Mixed English and 中文 content.",
        ]

        for test_text in test_texts:
            print(f"\n  测试文本: '{test_text[:50]}...'")

            if ZERO_AVAILABLE and tokenizer_zero:
                times = []
                for _ in range(100):
                    start = time.perf_counter()
                    _ = tokenizer_zero.encode(test_text)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)

                avg_time = sum(times) / len(times)
                print(f"    Zero Tokenizer: {avg_time*1000:.4f}ms")

            if HF_AVAILABLE and tokenizer_hf:
                times = []
                for _ in range(100):
                    start = time.perf_counter()
                    _ = tokenizer_hf.encode(test_text)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)

                avg_time = sum(times) / len(times)
                print(f"    HF Tokenizer: {avg_time*1000:.4f}ms")

    def benchmark_encoding_batch(self, tokenizer_zero=None, tokenizer_hf=None) -> None:
        """基准测试：批量编码速度"""
        print("\n" + "="*70)
        print(f"📊 批量编码速度测试 - {self.ALGORITHMS[self.algorithm]}")
        print("="*70)

        batch_sizes = [10, 100, 1000]

        for batch_size in batch_sizes:
            print(f"\n  批量大小: {batch_size}")
            test_batch = self.test_texts[:batch_size]

            if ZERO_AVAILABLE and tokenizer_zero:
                # 检查是否有encode_batch方法
                if hasattr(tokenizer_zero, 'encode_batch'):
                    start = time.perf_counter()
                    _ = tokenizer_zero.encode_batch(test_batch)
                    elapsed = time.perf_counter() - start
                    throughput = batch_size / elapsed
                    print(f"    Zero Tokenizer (并行): {throughput:.0f} 条/秒 ({elapsed*1000:.2f}ms)")
                else:
                    # 串行处理
                    start = time.perf_counter()
                    for text in test_batch:
                        _ = tokenizer_zero.encode(text)
                    elapsed = time.perf_counter() - start
                    throughput = batch_size / elapsed
                    print(f"    Zero Tokenizer (串行): {throughput:.0f} 条/秒 ({elapsed*1000:.2f}ms)")

            if HF_AVAILABLE and tokenizer_hf:
                start = time.perf_counter()
                _ = tokenizer_hf.encode_batch(test_batch)
                elapsed = time.perf_counter() - start
                throughput = batch_size / elapsed
                print(f"    HF Tokenizer: {throughput:.0f} 条/秒 ({elapsed*1000:.2f}ms)")

    def benchmark_decoding(self, tokenizer_zero=None, tokenizer_hf=None) -> None:
        """基准测试：解码速度"""
        print("\n" + "="*70)
        print(f"📊 解码速度测试 - {self.ALGORITHMS[self.algorithm]}")
        print("="*70)

        test_text = "The quick brown fox jumps over the lazy dog."

        if ZERO_AVAILABLE and tokenizer_zero:
            tokens = tokenizer_zero.encode(test_text)

            times = []
            for _ in range(100):
                start = time.perf_counter()
                _ = tokenizer_zero.decode(tokens)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_time = sum(times) / len(times)
            print(f"  ✅ Zero Tokenizer: {avg_time*1000:.4f}ms (平均)")

        if HF_AVAILABLE and tokenizer_hf:
            encoding = tokenizer_hf.encode(test_text)
            tokens = encoding.ids

            times = []
            for _ in range(100):
                start = time.perf_counter()
                _ = tokenizer_hf.decode(tokens)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_time = sum(times) / len(times)
            print(f"  ✅ HF Tokenizer: {avg_time*1000:.4f}ms (平均)")

    def run_all_benchmarks(self) -> None:
        """运行所有基准测试"""
        print("="*70)
        print(f"🚀 完整性能对比基准测试 - {self.ALGORITHMS[self.algorithm]}")
        print("="*70)
        print(f"算法: {self.ALGORITHMS[self.algorithm]}")
        print(f"词汇表大小: {self.vocab_size}")
        print(f"迭代次数: {self.iterations}")
        print(f"训练语料: {len(self.train_corpus)} 条")
        print(f"测试文本: {len(self.test_texts)} 条")

        # 1. 字典初始化测试（如果适用）
        self.benchmark_dict_init()

        # 2. 训练测试
        tokenizer_zero, tokenizer_hf = self.benchmark_training()

        # 3. 单条编码测试
        self.benchmark_encoding_single(tokenizer_zero, tokenizer_hf)

        # 4. 批量编码测试
        self.benchmark_encoding_batch(tokenizer_zero, tokenizer_hf)

        # 5. 解码测试
        self.benchmark_decoding(tokenizer_zero, tokenizer_hf)

    def save_results(self, output_file: str) -> None:
        """保存结果到JSON文件"""
        output_path = Path(__file__).parent / output_file

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Zero Tokenizer vs HuggingFace Tokenizers 完整性能对比"
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['bpe', 'bbpe', 'unigram', 'wordpiece', 'all'],
        default='bpe',
        help='测试的算法 (默认: bpe，all表示测试所有算法)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=1000,
        help='词汇表大小 (默认: 1000)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='训练迭代次数 (默认: 5)'
    )

    args = parser.parse_args()

    if not ZERO_AVAILABLE:
        print("\n❌ 错误: Zero Tokenizer未安装")
        print("   请先运行: maturin develop")
        sys.exit(1)

    if not HF_AVAILABLE:
        print("\n⚠️  警告: HuggingFace tokenizers未安装")
        print("   将只测试Zero Tokenizer性能")
        print("   要进行对比测试，请运行: pip install tokenizers")
        print()

    # 测试所有算法或单个算法
    algorithms_to_test = ['bpe', 'bbpe', 'unigram', 'wordpiece'] if args.algorithm == 'all' else [args.algorithm]

    for algo in algorithms_to_test:
        print("\n" + "█"*70)
        print(f"  开始测试: {ComprehensiveBenchmark.ALGORITHMS[algo]}")
        print("█"*70)

        benchmark = ComprehensiveBenchmark(
            algorithm=algo,
            vocab_size=args.vocab_size,
            iterations=args.iterations
        )
        benchmark.run_all_benchmarks()
        benchmark.save_results(f"benchmark_{algo}_results.json")

        print("\n" + "="*70)
        print(f"✅ {ComprehensiveBenchmark.ALGORITHMS[algo]} 测试完成！")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
