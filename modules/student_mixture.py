# student_mixture.py
# Author: mkw (reimplemented as StudentTMixture)
# Desc  : Simple Student-t Mixture Model (SMM) with GaussianMixture-like API
#Author  :   mkw 
#Time    :   2025/11/28 16:07:59
#Desc    :   None


import numpy as np
from math import lgamma


class StudentTMixture:
    """
    简单的 Student-t 混合模型（SMM），只支持对角协方差。
    接口尽量贴近 sklearn.mixture.GaussianMixture：
        - __init__(n_components=3, df=5.0, max_iter=100, tol=1e-3, ...)
        - fit(X)
        - predict_proba(X)
        - 属性: weights_, means_, covariances_

    参数:
        n_components: 混合分量数 K（你这边固定为 3）
        df         : 自由度 ν（尾越重 ν 越小，通常 3~10 比较合理）
        max_iter   : EM 最大迭代次数
        tol        : 对数似然收敛阈值
        reg_covar  : 对角协方差正则，防止数值为 0
        verbose    : 是否打印 log-likelihood
        random_state: 随机种子（int 或 None）

    注意:
        - 支持输入 X 为 (N,) 或 (N, D)，内部会统一成 (N, D)
        - 为了兼容你当前数据量较少的情况，当 N < n_components 时，
          会允许有放回的随机初始化（replace=True），避免随机下标报错。
    """

    def __init__(
        self,
        n_components=3,
        df=2.0,
        max_iter=100,
        tol=1e-3,
        reg_covar=1e-6,
        verbose=False,
        random_state=None,
    ):
        self.n_components = int(n_components)
        self.df = float(df)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.verbose = verbose

        # ✅ 始终使用 RandomState 实例，而不是 numpy.random 模块
        if isinstance(random_state, np.random.RandomState):
            # 已经是 RandomState 了，直接用
            self.random_state = random_state
        elif random_state is None:
            # 不传 seed，就新建一个随机的 RandomState
            self.random_state = np.random.RandomState()
        else:
            # 传了 int seed，就用该 seed 初始化
            self.random_state = np.random.RandomState(random_state)

        # 训练后会被填充：
        # self.weights_: (K,)
        # self.means_  : (K, D)
        # self.covariances_: (K, D)

    # -------- 参数初始化 --------
    def _init_params(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        N, D = X.shape

        # 混合权重初始化为均匀
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)

        # 均值初始化：随机挑 K 个样本（若样本数 < K，则允许有放回）
        replace_flag = N < self.n_components
        indices = self.random_state.choice(N, self.n_components, replace=replace_flag)
        self.means_ = X[indices].copy()  # (K, D)

        # 对角协方差初始化：用整体方差
        var = X.var(axis=0) + self.reg_covar
        if var.ndim == 0:
            var = np.array([var], dtype=np.float64)
        self.covariances_ = np.tile(var, (self.n_components, 1))  # (K, D)

    # -------- 计算每个分量的 t 分布 logpdf 和 δ_ik --------
    def _estimate_log_pdf_and_delta(self, X):
        """
        返回:
            log_pdf: (N, K)，第 i 行第 k 列是 log p(x_i | 分量 k)
            delta : (N, K)，对应马氏距离平方 δ_ik
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        N, D = X.shape
        K = self.n_components

        log_pdf = np.zeros((N, K), dtype=np.float64)
        delta = np.zeros((N, K), dtype=np.float64)
        nu = self.df

        for k in range(K):
            mean = self.means_[k]        # (D,)
            cov = self.covariances_[k]   # (D,)
            diff = X - mean              # (N, D)
            inv_cov = 1.0 / cov          # 对角协方差的逆

            # Mahalanobis 距离平方 δ_ik = (x - μ)^T Σ^{-1} (x - μ)
            delta_k = np.sum(diff ** 2 * inv_cov, axis=1)  # (N,)
            delta[:, k] = delta_k

            # log |Σ|
            log_det = np.sum(np.log(cov))

            # 多维 Student-t 的归一化常数 log c
            # log c = lgamma((ν + D)/2) - lgamma(ν/2) - 0.5 * (D * log(νπ) + log|Σ|)
            log_c = (
                lgamma((nu + D) / 2.0)
                - lgamma(nu / 2.0)
                - 0.5 * (D * np.log(nu * np.pi) + log_det)
            )

            # log pdf = log c - 0.5 * (ν + D) * log(1 + δ_ik / ν)
            log_pdf[:, k] = log_c - 0.5 * (nu + D) * np.log1p(delta_k / nu)

        return log_pdf, delta

    # -------- E-step --------
    def _e_step(self, X):
        """
        返回:
            resp: 责任度 γ_ik, (N, K)
            w   : 缩放因子 w_ik = (ν + D) / (ν + δ_ik), (N, K)
            ll  : 当前 log-likelihood
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        N, D = X.shape

        log_pdf, delta = self._estimate_log_pdf_and_delta(X)
        # log p(x_i, k) = log π_k + log p(x_i | k)
        log_prob = log_pdf + np.log(self.weights_)

        # log-sum-exp 做归一化
        max_log = log_prob.max(axis=1, keepdims=True)
        log_prob_norm = max_log + np.log(
            np.sum(np.exp(log_prob - max_log), axis=1, keepdims=True)
        )
        log_resp = log_prob - log_prob_norm
        resp = np.exp(log_resp)  # γ_ik

        # 期望缩放因子 w_ik
        nu = self.df
        w = (nu + D) / (nu + delta)

        # 总 log-likelihood
        ll = log_prob_norm.sum()
        return resp, w, ll

    # -------- M-step --------
    def _m_step(self, X, resp, w):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        N, D = X.shape
        K = self.n_components

        # 有效样本数 N_k
        N_k = resp.sum(axis=0)  # (K,)
        # 避免除以 0
        N_k_safe = N_k + 1e-12
        self.weights_ = N_k_safe / (N_k_safe.sum() + 1e-12)

        means = np.zeros((K, D), dtype=np.float64)
        covs = np.zeros((K, D), dtype=np.float64)

        for k in range(K):
            # r_ik * w_ik
            rw = resp[:, k] * w[:, k]  # (N,)
            denom = rw.sum() + 1e-12

            # 均值 μ_k = Σ_i r_ik w_ik x_i / Σ_i r_ik w_ik
            means[k] = (rw[:, None] * X).sum(axis=0) / denom

            # 协方差 Σ_k（对角）
            diff = X - means[k]  # (N, D)
            covs[k] = (
                (rw[:, None] * (diff ** 2)).sum(axis=0) / (N_k_safe[k])
                + self.reg_covar
            )

        self.means_ = means
        self.covariances_ = covs

    # -------- 对外接口：fit --------
    def fit(self, X):
        """
        X: (N,) 或 (N, D) 的 numpy 数组
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]

        self._init_params(X)
        prev_ll = None

        for it in range(self.max_iter):
            resp, w, ll = self._e_step(X)
            self._m_step(X, resp, w)

            if self.verbose:
                print(f"[SMM] iter {it}, log-likelihood = {ll:.4f}")

            if prev_ll is not None and abs(ll - prev_ll) < self.tol * (abs(ll) + 1.0):
                break
            prev_ll = ll

        return self

    # -------- 对外接口：predict_proba --------
    def predict_proba(self, X):
        """
        返回责任度 γ_ik，形状 (N, K)，
        行为与 GaussianMixture.predict_proba 类似。
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]

        log_pdf, _ = self._estimate_log_pdf_and_delta(X)
        log_prob = log_pdf + np.log(self.weights_)

        max_log = log_prob.max(axis=1, keepdims=True)
        log_prob_norm = max_log + np.log(
            np.sum(np.exp(log_prob - max_log), axis=1, keepdims=True)
        )
        log_resp = log_prob - log_prob_norm
        resp = np.exp(log_resp)
        return resp
