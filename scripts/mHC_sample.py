import numpy as np


def print_matrix(name: str, x: np.ndarray, decimals: int = 4) -> None:
    """格式化打印矩阵。"""
    print(f"\n{name}:")
    print(np.array2string(x, precision=decimals, suppress_small=True))


def print_stats(name: str, matrix: np.ndarray, output: np.ndarray) -> None:
    """打印混合矩阵的行和、列和，以及输出结果的范数。"""
    print(f"\n{name} row sums:", np.round(matrix.sum(axis=1), 4))
    print(f"{name} col sums:", np.round(matrix.sum(axis=0), 4))
    print(f"{name} output norm:", round(np.linalg.norm(output), 4))


def sinkhorn_to_doubly_stochastic(
    matrix: np.ndarray,
    steps: int = 30,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    一个用于演示的 Sinkhorn 归一化函数。

    它会反复做两件事：
    1. 把每一行归一化，让每一行加起来接近 1；
    2. 把每一列归一化，让每一列加起来接近 1。

    迭代多轮之后，矩阵会接近“双随机矩阵”：
    - 每一行的和约等于 1；
    - 每一列的和约等于 1；
    - 所有元素非负。

    注意：
    这不是 DeepSeek V4 中 mHC 的完整实现。
    这里只是用来演示受约束的信息流混合这个过程，便于读者感受。
    """
    x = matrix.astype(np.float64).copy()

    for _ in range(steps):
        # 先按行归一化：控制每条输出路径接收的信息总量
        x = x / (x.sum(axis=1, keepdims=True) + eps)

        # 再按列归一化：控制每条输入路径被分配出去的信息总量
        x = x / (x.sum(axis=0, keepdims=True) + eps)

    return x


def main() -> None:
    # 假设现在有 3 条残差路径 / 信息流。
    # 每一行代表一条 residual stream。
    # 每条 stream 是一个 2 维 hidden state。
    #
    # stream_1: 在第 1 个维度上有一个很强的信号；
    # stream_2: 在第 2 个维度上有一个较弱的信号；
    # stream_3: 同时带有两个维度上的混合信号。
    h = np.array([
        [10.0, 0.0],
        [0.0, 2.0],
        [1.0, 1.0],
    ])

    print_matrix("原始残差连接： H", h)
    print("原始 H 范数:", round(np.linalg.norm(h), 4))

    # 1. 普通残差连接：
    # 可以把它理解成单位矩阵 I。
    # 每条信息流基本只沿自己的路径继续传递，不和其他路径混合。
    identity_mix = np.eye(3)
    h_identity = identity_mix @ h

    print_matrix("1) 原始残差混合矩阵 I", identity_mix)
    print_matrix("1) 原始残差输出 I @ H", h_identity)
    print_stats("I", identity_mix, h_identity)

    # 2. HC-like 自由混合：
    # 允许多条 residual stream 之间互相混合。
    # 这样表达力更强，但混合矩阵 A 没有约束。
    # 某些信息就可能被重复叠加、放大或扭曲。
    free_mix = np.array([
        [1.8, 0.3, 0.2],
        [0.7, 1.5, 0.4],
        [0.5, 0.6, 1.7],
    ])

    h_free = free_mix @ h

    print_matrix("2) HC-like 自由混合矩阵 A", free_mix)
    print_matrix("2) HC-like 输出 A @ H", h_free)
    print_stats("A", free_mix, h_free)

    # 3. mHC-like 受约束混合：
    # 用 Sinkhorn 归一化，把刚才的自由矩阵 A
    # 转成一个近似双随机矩阵 P。
    #
    # P 仍然允许多路径混合，
    # 但每一行、每一列的和都接近 1。
    #
    # 这意味着：
    # - 每条输出路径接收到的信息总量更受控；
    # - 每条输入路径被分配出去的信息总量也更受控；
    # - 整体更像“重新分配信息流”，而不是任意放大信息流。
    constrained_mix = sinkhorn_to_doubly_stochastic(free_mix, steps=30)
    h_constrained = constrained_mix @ h

    print_matrix("3) mHC-like 受约束的矩阵 P，可以观察到每一行每一列加和均近似为1", constrained_mix)
    print_matrix("3) mHC-like 输出 P @ H，可以观察到幅度明显受控", h_constrained)
    print_stats("P", constrained_mix, h_constrained)

    print("- 普通残差连接：稳定，但每条信息流基本沿自己的路径传递。")
    print("- HC-like 自由混合：多路径表达力更强，但混合矩阵可能放大信号。")
    print("- mHC-like 受约束混合：仍然允许多路径混合，但行和列都接近 1。")

if __name__ == "__main__":
    main()