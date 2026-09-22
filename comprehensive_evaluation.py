#!/usr/bin/env python3
"""
comprehensive_evaluation.py

完整的实验评估框架 - 满足 CS245 项目评分标准
包含：
1. Baseline 对比
2. Ablation Studies（消融实验）
3. 所有 Benchmark Metrics（RMSE, MAE, Sentiment Alignment, HR@K）
4. 统计显著性检验
5. 可复现的实验设置
6. 详细的结果分析
"""

import argparse
import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================
# DeepSeek LLM 封装
# ============================

import requests

class DeepSeekEmbeddingModel:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-embedding"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def _api_embed(self, texts):
        try:
            url = f"{self.base_url}/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {"model": self.model, "input": texts}
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            print(f"❌ Embedding API 错误: {e}")
            return [np.zeros(768).tolist() for _ in texts]

    def embed_documents(self, texts):
        if not texts:
            return []
        return self._api_embed(texts)

    def embed_query(self, text):
        if not text:
            return np.zeros(768).tolist()
        return self._api_embed([text])[0]


class DeepSeekLLM:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1",
                 chat_model: str = "deepseek-chat", embedding_model: str = "deepseek-embedding",
                 seed: int = None):
        self.api_key = api_key
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.seed = seed
        self.usage = {"calls": 0, "errors": 0, "latency_seconds": 0.0}

    def _build_payload(self, messages, temperature, max_tokens):
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload

    def usage_stats(self) -> Dict:
        return dict(self.usage)

    def __call__(self, messages, temperature=0.7, max_tokens=800):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = self._build_payload(messages, temperature, max_tokens)
        started = time.time()
        self.usage["calls"] += 1
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            self.usage["errors"] += 1
            print(f"❌ Chat API 错误: {e}")
            return "（API 错误）"
        finally:
            self.usage["latency_seconds"] += time.time() - started

    def get_embedding_model(self):
        return DeepSeekEmbeddingModel(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.embedding_model
        )


# ============================
# 实验配置类
# ============================

class ExperimentConfig:
    """实验配置"""
    def __init__(self, name: str, enable_reflection: bool, use_memory: bool,
                 max_reference_reviews: int, description: str = "",
                 include_context: bool = True, agent_kind: str = "llm"):
        self.name = name
        self.enable_reflection = enable_reflection
        self.use_memory = use_memory
        self.max_reference_reviews = max_reference_reviews
        self.description = description
        self.include_context = include_context
        self.agent_kind = agent_kind

    def __str__(self):
        return (
            f"{self.name}: reflection={self.enable_reflection}, "
            f"memory={self.use_memory}, refs={self.max_reference_reviews}, "
            f"context={self.include_context}, agent={self.agent_kind}"
        )


# ============================
# 评估指标计算
# ============================

def extract_prediction(output) -> float:
    """Return the predicted stars from a simulator output, or None."""
    if not output or not isinstance(output, dict):
        return None
    if "output" in output and isinstance(output["output"], dict):
        value = output["output"].get("stars")
    else:
        value = output.get("stars")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_per_task_records(outputs: List[Dict],
                           groundtruths: List[Dict]) -> List[Dict]:
    """Pair predictions with ground truth into per-task error records."""
    records = []
    for index, (output, groundtruth) in enumerate(zip(outputs, groundtruths)):
        if not groundtruth or "stars" not in groundtruth:
            continue
        prediction = extract_prediction(output)
        actual = float(groundtruth["stars"])
        record = {
            "index": index,
            "predicted": prediction,
            "actual": actual,
            "error": None,
            "squared_error": None,
        }
        if prediction is not None:
            record["error"] = abs(prediction - actual)
            record["squared_error"] = (prediction - actual) ** 2
        records.append(record)
    return records


def calculate_additional_metrics(outputs: List[Dict], groundtruths: List[Dict]) -> Dict[str, float]:
    """
    计算额外的评估指标（补充 simulator.evaluate()）
    """
    records = build_per_task_records(outputs, groundtruths)
    valid = [
        record for record in records
        if (
            record["error"] is not None
            and record["predicted"] is not None
            and np.isfinite(record["predicted"])
            and np.isfinite(record["actual"])
        )
    ]
    if not valid:
        return {"error": "No valid predictions"}

    predicted_stars = np.array([record["predicted"] for record in valid])
    actual_stars = np.array([record["actual"] for record in valid])
    errors = predicted_stars - actual_stars

    metrics = {}
    metrics["rmse"] = float(np.sqrt(np.mean(np.square(errors))))
    metrics["mae"] = float(np.mean(np.abs(errors)))
    metrics["accuracy_exact"] = np.mean(predicted_stars == actual_stars)
    metrics["accuracy_±0.5"] = np.mean(np.abs(predicted_stars - actual_stars) <= 0.5)
    metrics["accuracy_±1.0"] = np.mean(np.abs(predicted_stars - actual_stars) <= 1.0)

    # 分布指标
    metrics["pred_mean"] = float(np.mean(predicted_stars))
    metrics["pred_std"] = float(np.std(predicted_stars))
    metrics["actual_mean"] = float(np.mean(actual_stars))
    metrics["actual_std"] = float(np.std(actual_stars))

    # 相关性
    if len(predicted_stars) > 1:
        correlation = np.corrcoef(predicted_stars, actual_stars)[0, 1]
        metrics["pearson_correlation"] = float(correlation)

    metrics["num_valid_predictions"] = len(valid)
    return metrics


def paired_bootstrap_test(records_a: List[Dict], records_b: List[Dict],
                          metric: str = "error", n_boot: int = 5000,
                          seed: int = 0) -> Dict:
    """Paired bootstrap test over per-task records.

    The two record lists must share the same task ordering and length.
    The difference is computed as mean(metric_a) - mean(metric_b), so a
    negative value means ``records_a`` has smaller errors.
    """
    if len(records_a) != len(records_b):
        raise ValueError("record lists must have the same length")
    if not records_a:
        raise ValueError("record lists must not be empty")

    try:
        diffs = np.array([
            float(a[metric]) - float(b[metric])
            for a, b in zip(records_a, records_b)
        ])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot compute metric '{metric}': {exc}") from exc

    rng = np.random.default_rng(seed)
    n = len(diffs)
    samples = rng.choice(diffs, size=(n_boot, n), replace=True)
    boot_means = samples.mean(axis=1)
    observed = float(diffs.mean())

    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    prob_positive = float(np.mean(boot_means >= 0))
    prob_negative = float(np.mean(boot_means <= 0))
    p_value = min(1.0, 2 * min(prob_positive, prob_negative))

    return {
        "metric": metric,
        "n_tasks": n,
        "mean_difference": observed,
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "p_value": p_value,
        "n_boot": n_boot,
    }


# ============================
# 实验运行器
# ============================

class ExperimentRunner:
    """实验运行器 - 负责运行所有实验配置"""

    def __init__(self, data_dir: str, task_set: str, api_key: str,
                 num_tasks: int = 100, max_workers: int = 5,
                 output_dir: str = "results", chat_model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com/v1",
                 seed: int = None, task_dir: str = None,
                 groundtruth_dir: str = None):
        self.data_dir = data_dir
        self.task_set = task_set
        self.api_key = api_key
        self.num_tasks = num_tasks
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.chat_model = chat_model
        self.base_url = base_url
        self.seed = seed
        self.task_dir = task_dir
        self.groundtruth_dir = groundtruth_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{self.run_id}")
        self.results = {}
        self.per_task_records = {}

    def run_experiment(self, config: ExperimentConfig) -> Dict:
        """运行单个实验配置"""
        # Imported lazily so --dry-run works without the simulator framework.
        from websocietysimulator import Simulator
        from improved_agent_with_quality import (
            DeterministicSimulationAgent,
            ImprovedSimulationAgent,
        )

        print(f"\n{'='*80}")
        print(f"🧪 运行实验: {config.name}")
        print(f"{'='*80}")
        print(f"配置: {config}")

        # 初始化模拟器
        simulator = Simulator(
            data_dir=self.data_dir,
            device="cpu",
            cache=True
        )

        # 加载任务（默认约定 example/track1/<task-set>/，可用 CLI 覆盖）
        simulator.set_task_and_groundtruth(
            task_dir=(
                self.task_dir
                or f"example/track1/{self.task_set}/tasks"
            ),
            groundtruth_dir=(
                self.groundtruth_dir
                or f"example/track1/{self.task_set}/groundtruth"
            )
        )

        # 配置 Agent
        if config.agent_kind == "deterministic":
            agent_class = DeterministicSimulationAgent
        else:
            class ConfiguredAgent(ImprovedSimulationAgent):
                def __init__(self, llm):
                    super().__init__(
                        llm=llm,
                        enable_reflection=config.enable_reflection,
                        use_memory=config.use_memory,
                        max_reference_reviews=config.max_reference_reviews,
                        include_context=config.include_context
                    )

            agent_class = ConfiguredAgent

        llm_client = DeepSeekLLM(
            api_key=self.api_key,
            base_url=self.base_url,
            chat_model=self.chat_model,
            seed=self.seed
        )
        simulator.set_agent(agent_class)
        simulator.set_llm(llm_client)

        # 运行模拟
        print(f"\n⚙️  运行 {self.num_tasks} 个任务...")
        start_time = time.time()

        outputs = simulator.run_simulation(
            number_of_tasks=self.num_tasks,
            enable_threading=True,
            max_workers=self.max_workers
        )

        elapsed_time = time.time() - start_time

        # 逐任务结果持久化
        groundtruths = self._extract_groundtruths(simulator)
        records = build_per_task_records(outputs, groundtruths)
        self.per_task_records[config.name] = records
        self._save_per_task_records(config.name, records)

        print(f"✅ 完成！用时: {elapsed_time:.2f}秒")
        print(f"   平均每任务: {elapsed_time/self.num_tasks:.2f}秒")

        # Calculate project-owned metrics before calling the framework evaluator.
        # The framework may truncate or mutate its output list during evaluation.
        try:
            additional_metrics = calculate_additional_metrics(
                outputs, groundtruths
            )
        except Exception as e:
            additional_metrics = {"metric_error": str(e)}
            print(f"⚠️ 无法计算自有指标: {e}")

        # 评估
        print("\n📊 评估中...")
        eval_results = {}
        evaluation_warning = None
        try:
            framework_result = simulator.evaluate()

            # websocietysimulator==1.0.0a30 returns (metrics, error_log),
            # while other versions may return the metrics dictionary directly.
            if isinstance(framework_result, tuple):
                eval_results, framework_error_log = framework_result
                if not isinstance(eval_results, dict):
                    raise TypeError(
                        "simulator.evaluate() tuple must start with a dict"
                    )
                if framework_error_log:
                    eval_results["framework_error_log"] = framework_error_log
            else:
                eval_results = framework_result

            if not isinstance(eval_results, dict):
                raise TypeError(
                    "simulator.evaluate() must return a dict or (dict, error_log)"
                )

        except Exception as e:
            evaluation_warning = str(e)
            print(f"⚠️ 框架评估失败，保留自有指标: {e}")
            import traceback
            traceback.print_exc()

        if not isinstance(eval_results, dict):
            eval_results = {}

        # Project-owned RMSE/MAE remain authoritative even when the framework
        # evaluator succeeds, because the framework has different metric names
        # and can fail on all-real or partial task sets.
        eval_results.update(additional_metrics)

        if evaluation_warning:
            eval_results["evaluation_warning"] = evaluation_warning

        # 添加元数据
        eval_results["config"] = {
            "name": config.name,
            "enable_reflection": config.enable_reflection,
            "use_memory": config.use_memory,
            "max_reference_reviews": config.max_reference_reviews,
            "include_context": config.include_context,
            "agent_kind": config.agent_kind
        }
        eval_results["num_tasks"] = self.num_tasks
        eval_results["elapsed_time"] = elapsed_time
        eval_results["timestamp"] = datetime.now().isoformat()
        eval_results["llm_usage"] = llm_client.usage_stats()

        return eval_results

    def run_all_experiments(self, configs: List[ExperimentConfig]):
        """运行所有实验配置"""
        print("\n" + "🚀"*40)
        print("开始运行完整实验套件")
        print("🚀"*40 + "\n")

        self._save_run_metadata(configs)

        for config in configs:
            try:
                results = self.run_experiment(config)
                self.results[config.name] = results

                # 保存中间结果
                self._save_intermediate_results(config.name)

            except Exception as e:
                print(f"❌ 实验 {config.name} 失败: {e}")
                import traceback
                traceback.print_exc()
                self.results[config.name] = {"error": str(e)}

        print("\n" + "✅"*40)
        print("所有实验完成！")
        print("✅"*40 + "\n")

        return self.results

    def _save_intermediate_results(self, config_name: str):
        """保存中间结果"""
        os.makedirs(self.run_dir, exist_ok=True)
        filename = os.path.join(self.run_dir, f"results_{config_name}.json")

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results[config_name], f, indent=4, ensure_ascii=False)

        print(f"💾 中间结果已保存: {filename}")

    def _extract_groundtruths(self, simulator) -> List[Dict]:
        """兼容不同版本的 simulator groundtruth 属性名"""
        for attribute in ('groundtruth_data', 'groundtruth_pool', 'groundtruths'):
            if hasattr(simulator, attribute):
                return getattr(simulator, attribute)[:self.num_tasks]
        return []

    def _save_per_task_records(self, config_name: str, records: List[Dict]):
        """保存逐任务预测与误差"""
        os.makedirs(self.run_dir, exist_ok=True)
        filename = os.path.join(self.run_dir, f"per_task_{config_name}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4, ensure_ascii=False)
        print(f"💾 逐任务结果已保存: {filename} ({len(records)} 条)")

    def build_run_metadata(self, configs: List[ExperimentConfig]) -> Dict:
        """记录运行级元数据，保证实验可追溯"""
        return {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "data_dir": self.data_dir,
            "task_set": self.task_set,
            "num_tasks": self.num_tasks,
            "max_workers": self.max_workers,
            "chat_model": self.chat_model,
            "base_url": self.base_url,
            "seed": self.seed,
            "task_dir": self.task_dir,
            "groundtruth_dir": self.groundtruth_dir,
            "output_dir": self.output_dir,
            "experiments": [config.name for config in configs],
        }

    def _save_run_metadata(self, configs: List[ExperimentConfig]):
        os.makedirs(self.run_dir, exist_ok=True)
        filename = os.path.join(self.run_dir, "metadata.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(
                self.build_run_metadata(configs), f,
                indent=4, ensure_ascii=False
            )
        print(f"💾 运行元数据已保存: {filename}")

    def compare(self, name_a: str, name_b: str, metric: str = "error") -> Dict:
        """对两个配置做配对 bootstrap 检验（需要同一批任务顺序）"""
        if (
            name_a not in self.per_task_records
            or name_b not in self.per_task_records
        ):
            raise KeyError(f"missing per-task records for {name_a} or {name_b}")
        return paired_bootstrap_test(
            self.per_task_records[name_a],
            self.per_task_records[name_b],
            metric=metric,
            seed=self.seed or 0,
        )


# ============================
# 结果分析器
# ============================

class ResultsAnalyzer:
    """结果分析器 - 生成对比表格、图表、统计分析"""

    def __init__(self, results: Dict[str, Dict]):
        self.results = results

    def generate_comparison_table(self) -> str:
        """生成对比表格（Markdown格式）"""
        table = "\n## 📊 实验结果对比表\n\n"
        table += "| 配置 | RMSE | MAE | Sentiment Acc | Accuracy(±0.5) | Correlation | 时间(s) |\n"
        table += "|------|------|-----|---------------|----------------|-------------|----------|\n"

        for name, result in self.results.items():
            if "error" in result:
                table += f"| {name} | ERROR | - | - | - | - | - |\n"
                continue

            rmse = result.get("rmse", "N/A")
            mae = result.get("mae", "N/A")
            sent = result.get("sentiment_alignment", "N/A")
            acc = result.get("accuracy_±0.5", "N/A")
            corr = result.get("pearson_correlation", "N/A")
            time_val = result.get("elapsed_time", "N/A")

            # 格式化数值
            rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else rmse
            mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else mae
            sent_str = f"{sent:.4f}" if isinstance(sent, (int, float)) else sent
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else acc
            corr_str = f"{corr:.4f}" if isinstance(corr, (int, float)) else corr
            time_str = f"{time_val:.1f}" if isinstance(time_val, (int, float)) else time_val

            table += f"| {name} | {rmse_str} | {mae_str} | {sent_str} | {acc_str} | {corr_str} | {time_str} |\n"

        return table

    def generate_ablation_analysis(self, baseline_name: str) -> str:
        """生成消融分析"""
        if baseline_name not in self.results:
            return "\n⚠️ 未找到 baseline 结果\n"

        baseline = self.results[baseline_name]
        analysis = "\n## 🔬 Ablation Study 分析\n\n"

        for name, result in self.results.items():
            if name == baseline_name or "error" in result:
                continue

            analysis += f"\n### {name} vs {baseline_name}\n\n"

            # 计算各项指标的改进
            metrics = ["rmse", "mae", "sentiment_alignment", "accuracy_±0.5"]

            for metric in metrics:
                if metric in result and metric in baseline:
                    base_val = baseline[metric]
                    exp_val = result[metric]

                    # RMSE/MAE 越小越好，其他越大越好
                    if metric in ["rmse", "mae"]:
                        improvement = (base_val - exp_val) / base_val * 100
                        symbol = "↓" if exp_val < base_val else "↑"
                    else:
                        improvement = (exp_val - base_val) / base_val * 100
                        symbol = "↑" if exp_val > base_val else "↓"

                    analysis += f"- **{metric}**: {exp_val:.4f} (baseline: {base_val:.4f}) "
                    analysis += f"→ {symbol} {abs(improvement):.2f}%\n"

        return analysis

    def generate_statistical_analysis(self) -> str:
        """生成统计分析（全部实验失败时给出说明而不是崩溃）"""
        analysis = "\n## 📈 统计分析\n\n"

        valid = {
            name: result
            for name, result in self.results.items()
            if "error" not in result and "rmse" in result
        }
        if not valid:
            failed = [
                name for name, result in self.results.items() if "error" in result
            ]
            analysis += "没有可用的实验结果。\n"
            if failed:
                analysis += f"- 失败的配置: {', '.join(failed)}\n"
            return analysis

        # 找到最好的配置
        best_rmse = min(
            (r.get("rmse", float('inf')), name) for name, r in valid.items()
        )
        best_mae = min(
            (r.get("mae", float('inf')), name) for name, r in valid.items()
        )
        best_sent = max(
            (r.get("sentiment_alignment", 0), name) for name, r in valid.items()
        )

        analysis += "### 最佳配置\n\n"
        analysis += f"- **最低 RMSE**: {best_rmse[1]} ({best_rmse[0]:.4f})\n"
        analysis += f"- **最低 MAE**: {best_mae[1]} ({best_mae[0]:.4f})\n"
        analysis += f"- **最高 Sentiment Alignment**: {best_sent[1]} ({best_sent[0]:.4f})\n"

        return analysis

    def save_full_report(self, filename: str = "experiment_report.md"):
        """保存完整报告"""
        report = "# CS245 Track 1 - 实验评估完整报告\n\n"
        report += f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # 实验概述
        first_result = next(iter(self.results.values()), {})
        report += "## 📋 实验概述\n\n"
        report += f"- **总实验数**: {len(self.results)}\n"
        report += "- **数据集**: Yelp\n"
        report += f"- **每个实验的任务数**: {first_result.get('num_tasks', 'N/A')}\n\n"

        # 添加各个分析部分
        report += self.generate_comparison_table()
        report += self.generate_ablation_analysis("Baseline")
        report += self.generate_statistical_analysis()

        # 保存
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📄 完整报告已保存: {filename}")
        return filename


# ============================
# CLI
# ============================

def default_experiments() -> List[ExperimentConfig]:
    """默认实验配置：确定性/无上下文 baseline + full + 三个消融"""
    return [
        ExperimentConfig(
            name="Deterministic",
            enable_reflection=False,
            use_memory=False,
            max_reference_reviews=0,
            description="Deterministic baseline: user's historical average",
            agent_kind="deterministic"
        ),
        ExperimentConfig(
            name="Baseline",
            enable_reflection=False,
            use_memory=False,
            max_reference_reviews=3,
            description="Simple baseline without reflection or memory"
        ),
        ExperimentConfig(
            name="No_Context",
            enable_reflection=False,
            use_memory=False,
            max_reference_reviews=0,
            description="LLM baseline without profile or reference reviews",
            include_context=False
        ),
        ExperimentConfig(
            name="Full",
            enable_reflection=True,
            use_memory=True,
            max_reference_reviews=5,
            description="Full model with all features enabled"
        ),
        ExperimentConfig(
            name="No_Reflection",
            enable_reflection=False,
            use_memory=True,
            max_reference_reviews=5,
            description="Ablation: Remove reflection"
        ),
        ExperimentConfig(
            name="No_Memory",
            enable_reflection=True,
            use_memory=False,
            max_reference_reviews=5,
            description="Ablation: Remove memory"
        ),
        ExperimentConfig(
            name="Fewer_References",
            enable_reflection=True,
            use_memory=True,
            max_reference_reviews=2,
            description="Ablation: Reduce reference reviews to 2"
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    """CLI 参数定义"""
    parser = argparse.ArgumentParser(
        description="Run the CS245 Track A ablation suite."
    )
    parser.add_argument(
        "--data-dir", default="Dataset",
        help="Simulator dataset root directory (default: Dataset)."
    )
    parser.add_argument(
        "--task-dir", default=None,
        help="Task directory; defaults to example/track1/<task-set>/tasks."
    )
    parser.add_argument(
        "--groundtruth-dir", default=None,
        help="Groundtruth directory; defaults to "
             "example/track1/<task-set>/groundtruth."
    )
    parser.add_argument(
        "--task-set", default="yelp",
        choices=("yelp", "amazon", "goodreads"),
        help="Task/groundtruth set under example/track1/ (default: yelp)."
    )
    parser.add_argument(
        "--num-tasks", type=int, default=10,
        help="Number of tasks per experiment (default: 10)."
    )
    parser.add_argument(
        "--max-workers", type=int, default=5,
        help="Threading workers passed to run_simulation (default: 5)."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Optional LLM sampling seed, recorded with the run."
    )
    parser.add_argument(
        "--api-key", default=None,
        help="LLM API key; falls back to DEEPSEEK_API_KEY."
    )
    parser.add_argument(
        "--chat-model", default="deepseek-chat",
        help="Chat model name (default: deepseek-chat)."
    )
    parser.add_argument(
        "--base-url", default="https://api.deepseek.com/v1",
        help="LLM API base URL."
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory for reports and JSON results (default: results)."
    )
    parser.add_argument(
        "--experiment", nargs="*", default=None,
        help="Experiment names to run (default: all)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate settings and exit without importing the framework "
             "or calling the LLM."
    )
    return parser


def print_dry_run(args, experiments: List[ExperimentConfig],
                  api_key_configured: bool) -> None:
    """打印将要执行的配置，不触发任何 API 调用"""
    print("DRY RUN - no LLM calls will be made")
    print(f"  data_dir    : {args.data_dir}")
    print(f"  task_set    : {args.task_set}")
    print(f"  num_tasks   : {args.num_tasks}")
    print(f"  max_workers : {args.max_workers}")
    print(f"  chat_model  : {args.chat_model}")
    print(f"  base_url    : {args.base_url}")
    print(f"  output_dir  : {args.output_dir}")
    print(f"  api_key     : {'configured' if api_key_configured else 'missing'}")
    print(f"  experiments : {len(experiments)}")
    for config in experiments:
        print(f"    - {config.name}: {config}")


def main(argv=None) -> int:
    """主函数 - 解析 CLI 并运行实验套件"""
    parser = build_parser()
    args = parser.parse_args(argv)

    experiments = default_experiments()
    if args.experiment:
        known = {config.name for config in experiments}
        unknown = [name for name in args.experiment if name not in known]
        if unknown:
            parser.error(
                f"unknown experiment(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(known))}"
            )
        selected = set(args.experiment)
        experiments = [c for c in experiments if c.name in selected]

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")

    if args.dry_run:
        print_dry_run(args, experiments, api_key_configured=bool(api_key))
        return 0

    if not api_key:
        print("ERROR: no API key. Pass --api-key or set DEEPSEEK_API_KEY.")
        return 2

    runner = ExperimentRunner(
        data_dir=args.data_dir,
        task_set=args.task_set,
        api_key=api_key,
        num_tasks=args.num_tasks,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        chat_model=args.chat_model,
        base_url=args.base_url,
        seed=args.seed,
        task_dir=args.task_dir,
        groundtruth_dir=args.groundtruth_dir
    )

    results = runner.run_all_experiments(experiments)
    os.makedirs(runner.run_dir, exist_ok=True)

    # ============================================
    # 分析结果
    # ============================================

    analyzer = ResultsAnalyzer(results)

    report_file = analyzer.save_full_report(
        filename=os.path.join(runner.run_dir, "experiment_report.md")
    )

    print("\n" + "="*80)
    print("📊 实验总结")
    print("="*80)
    print(analyzer.generate_comparison_table())
    print(analyzer.generate_statistical_analysis())

    comparisons = {}
    for baseline_name in ("Deterministic", "Baseline"):
        if (
            "Full" in runner.per_task_records
            and baseline_name in runner.per_task_records
        ):
            key = f"Full_vs_{baseline_name}"
            comparisons[key] = runner.compare("Full", baseline_name)
            print(f"\n📈 配对 bootstrap（{key}, metric=error）:")
            print(json.dumps(comparisons[key], indent=4, ensure_ascii=False))

    if comparisons:
        comparison_file = os.path.join(runner.run_dir, "comparisons.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparisons, f, indent=4, ensure_ascii=False)
        print(f"💾 显著性检验已保存: {comparison_file}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(runner.run_dir, f"all_results_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n💾 原始结果已保存: {json_file}")

    print("\n" + "🎉"*40)
    print("实验评估完成！")
    print("🎉"*40)
    print(f"\n📝 报告: {report_file}")
    return 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    sys.exit(main())
