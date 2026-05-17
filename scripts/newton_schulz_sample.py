import numpy as np


def ns_polar_orthogonalize(x: np.ndarray, steps: int = 3) -> np.ndarray:
    """
    X_{k+1} = 1/2 * X_k @ (3I - X_k.T @ X_k)

    教学 toy 版本，不是完整 Muon 工程实现。
    """
    x = x.astype(np.float64)
    smax = np.linalg.svd(x, compute_uv=False)[0]
    x = x / smax

    eye = np.eye(x.shape[1])

    for _ in range(steps):
        x = 0.5 * x @ (3 * eye - x.T @ x)

    return x


def main() -> None:
    mt = np.array([
        [0.0, 0.0, 3.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
    ])

    print("原始矩阵：")
    print(mt)
    print("\n逐轮正交化结果：")

    print("\n第一轮：")
    print(ns_polar_orthogonalize(mt, steps=1))
    print("\n第二轮：")
    print(ns_polar_orthogonalize(mt, steps=2))
    print("\n第三轮：")
    print(ns_polar_orthogonalize(mt, steps=3))
    print("\n第四轮：")
    print(ns_polar_orthogonalize(mt, steps=4))
    print("\n第五轮：")
    print(ns_polar_orthogonalize(mt, steps=5))

    real_mt = np.array([
        [7, 24, 4],
        [58, 6, 9],
        [12, 15, 47],
    ])

    print("\n能看出方向性的稍复杂矩阵：")
    print(real_mt)

    print("\n五轮后结果：")
    print(ns_polar_orthogonalize(real_mt, steps=5))


if __name__ == "__main__":
    main()
