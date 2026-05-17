import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 定义函数
# =========================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softplus(x):
    # 数值稳定版 softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def sqrt_softplus(x):
    return np.sqrt(softplus(x))

# sigmoid 的导数
def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

# sqrt(softplus(x)) 的导数
# softplus'(x) = sigmoid(x)
# d/dx sqrt(softplus(x)) = sigmoid(x) / (2 * sqrt(softplus(x)))
def sqrt_softplus_grad(x):
    sp = softplus(x)
    return sigmoid(x) / (2 * np.sqrt(sp))


# =========================
# 2. 画函数曲线
# =========================

x = np.linspace(-10, 15, 1000)
y_sigmoid = sigmoid(x)
y_sqrt_sp = sqrt_softplus(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y_sigmoid, label="sigmoid(x)")
plt.plot(x, y_sqrt_sp, label="sqrt(softplus(x))")
plt.axvline(0, linestyle="--", alpha=0.5)
plt.axhline(0, linestyle="--", alpha=0.5)
plt.title("sigmoid(x) vs sqrt(softplus(x))")
plt.xlabel("x")
plt.ylabel("function value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =========================
# 3. 画导数曲线
# =========================

y_sigmoid_grad = sigmoid_grad(x)
y_sqrt_sp_grad = sqrt_softplus_grad(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y_sigmoid_grad, label="sigmoid'(x)")
plt.plot(x, y_sqrt_sp_grad, label="d/dx sqrt(softplus(x))")
plt.axvline(0, linestyle="--", alpha=0.5)
plt.axhline(0, linestyle="--", alpha=0.5)
plt.title("Gradient comparison")
plt.xlabel("x")
plt.ylabel("gradient")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =========================
# 4. 打印一些关键点，便于观察数值
# =========================

sample_points = np.array([-8, -4, -2, 0, 1, 2, 4, 8, 12], dtype=float)

print("关键点函数值对比：")
print(f"{'x':>8} | {'sigmoid(x)':>12} | {'sqrt(softplus(x))':>20}")
print("-" * 50)
for v in sample_points:
    print(f"{v:8.2f} | {sigmoid(v):12.6f} | {sqrt_softplus(v):20.6f}")

print("\n关键点导数对比：")
print(f"{'x':>8} | {'sigmoid_grad':>12} | {'sqrt_softplus_grad':>20}")
print("-" * 50)
for v in sample_points:
    print(f"{v:8.2f} | {sigmoid_grad(v):12.6f} | {sqrt_softplus_grad(v):20.6f}")


# =========================
# 5. top-k toy case
# =========================
# 假设某个 token 对一组 expert 的 raw score 如下
raw_scores = np.array([1.0, 2.0, 4.0, 5.0, 8.0, 12.0])
expert_names = [f"E{i}" for i in range(len(raw_scores))]

sigmoid_scores = sigmoid(raw_scores)
sqrt_sp_scores = sqrt_softplus(raw_scores)


def topk_normalized(scores, k=3):
    idx = np.argsort(scores)[::-1][:k]
    top_scores = scores[idx]
    norm_scores = top_scores / top_scores.sum()
    return idx, top_scores, norm_scores


k = 3
sig_idx, sig_top, sig_norm = topk_normalized(sigmoid_scores, k=k)
sp_idx, sp_top, sp_norm = topk_normalized(sqrt_sp_scores, k=k)

print("\nRaw scores:")
for name, score in zip(expert_names, raw_scores):
    print(f"{name}: {score:.4f}")

print("\nSigmoid -> Top-k normalized weights:")
for idx, score, weight in zip(sig_idx, sig_top, sig_norm):
    print(f"{expert_names[idx]}: transformed_score={score:.6f}, normalized_weight={weight:.6f}")

print("\nSqrt(Softplus) -> Top-k normalized weights:")
for idx, score, weight in zip(sp_idx, sp_top, sp_norm):
    print(f"{expert_names[idx]}: transformed_score={score:.6f}, normalized_weight={weight:.6f}")


# 为了便于可视化，把 top-k 权重画出来
sig_bar = np.zeros_like(raw_scores, dtype=float)
sp_bar = np.zeros_like(raw_scores, dtype=float)

for idx, weight in zip(sig_idx, sig_norm):
    sig_bar[idx] = weight

for idx, weight in zip(sp_idx, sp_norm):
    sp_bar[idx] = weight

x_pos = np.arange(len(raw_scores))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x_pos - width / 2, sig_bar, width=width, label="sigmoid top-k normalized")
plt.bar(x_pos + width / 2, sp_bar, width=width, label="sqrt(softplus) top-k normalized")
plt.xticks(x_pos, expert_names)
plt.title("Top-k normalized routing weights")
plt.xlabel("Experts")
plt.ylabel("Normalized weight")
plt.legend()
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()