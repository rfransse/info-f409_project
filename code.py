import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


#  payoff (player 1 reward)

# Actions: 0=Rock, 1=Paper, 2=Scissors
# Payoff: win=+1, loss=-1, tie=0
RPS_PAYOFF = np.array([
    [ 0, -1,  1],  # Rock
    [ 1,  0, -1],  # Paper
    [-1,  1,  0],  # Scissors
], dtype=float)

def sample_action(mix, rng):
    return rng.choice(3, p=mix)


def project_simplex(p, eps=1e-12):
    """Initial simple projection to simplex with small eps to avoid zeros."""
    p = np.maximum(p, eps)
    p = p / p.sum()
    return p


#  simplex grid (N=25 => 325 points) as in the paper

def simplex_grid_3(N=25):
    """
    Uniform grid for (pR, pP, pS) with step 1/(N-1).
    Total size = N(N+1)/2 = 325 for N=25.
    """
    step = 1.0 / (N - 1)
    pts = []
    for i in range(N):
        for j in range(N - i):
            pR = i * step
            pP = j * step
            pS = 1.0 - pR - pP
            pts.append(np.array([pR, pP, pS], dtype=float))
    return np.array(pts)  # (325,3)


#  -- Opponent estimators --

# EMA Equation 3
@dataclass
class EMAEstimator:
    mu: float = 0.005
    ybar: np.ndarray = None

    def reset(self):
        self.ybar = np.array([1/3, 1/3, 1/3], dtype=float)

    def update(self, opp_action):
        u = np.zeros(3, dtype=float)
        u[opp_action] = 1.0
        self.ybar = (1.0 - self.mu) * self.ybar + self.mu * u
        self.ybar = project_simplex(self.ybar)

    def estimate(self):
        return self.ybar

# Bayes Equations 4-5
@dataclass
class BayesEstimator:
    """
    BayesEstimator using equations 4-5 from the paper.
    """
    grid: np.ndarray
    beta: float = 0.995 # 1 - MU
    prior: np.ndarray = None
    loglik: np.ndarray = None

    def reset(self):
        M = len(self.grid)
        self.loglik = np.zeros(M, dtype=float)
        if self.prior is None:
            self.prior = np.ones(M, dtype=float) / M

    def update(self, opp_action, eps=1e-12):
        # Decay
        self.loglik *= self.beta
        # Add new
        self.loglik += np.log(np.clip(self.grid[:, opp_action], eps, 1.0))

    def posterior(self):
        # posterior P(y|H) ∝ prior(y) * exp(loglik(y))
        maxlog = np.max(self.loglik)
        unnorm = self.prior * np.exp(self.loglik - maxlog)
        Z = unnorm.sum()
        if Z <= 0:
            return np.ones_like(unnorm) / len(unnorm)
        return unnorm / Z

    def mean_estimate(self):
        p = self.posterior()
        return (p[:, None] * self.grid).sum(axis=0)


# IGA opponent (Equation 8)
@dataclass
class IGAOpponent:
    """
    Infinitesimal Gradient Ascent for player 2
    """
    eta: float = 0.05
    y: np.ndarray = None

    def reset(self, rng):
        self.y = project_simplex(rng.random(3))

    def step(self, x):
        grad_u2 = -(RPS_PAYOFF.T @ x)
        self.y = project_simplex(self.y + self.eta * grad_u2)
        return self.y

# PHC opponent 
@dataclass
class PHCOpponent:
    """
    Policy Hill-Climbing opponent. (not a lot of info in the paper)
    """
    alpha_q: float = 0.05
    delta: float = 0.002
    epsilon: float = 0.0
    q: np.ndarray = None
    pi: np.ndarray = None

    def reset(self, rng):
        self.q = np.zeros(3, dtype=float)
        self.pi = project_simplex(rng.random(3))

    def act(self, rng):
        if self.epsilon > 0 and rng.random() < self.epsilon:
            return rng.integers(0, 3)
        return rng.choice(3, p=self.pi)

    def update(self, action, reward):
        # zero-sum : reward2 = -reward1
        self.q[action] = (1 - self.alpha_q) * self.q[action] + self.alpha_q * reward

        # hill-climb policy
        best = int(np.argmax(self.q))
        for a in range(3):
            if a == best:
                self.pi[a] += self.delta
            else:
                self.pi[a] -= self.delta / 2.0
        self.pi = project_simplex(self.pi)

    def strategy(self):
        return self.pi


#  Hyper-Q learner (Equations 1-2)
class HyperQ:
    def __init__(self, grid, alpha=0.01, gamma=0.9, rng=None):
        self.grid = grid
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng() if rng is None else rng

        M = len(grid)
        # Q(y_idx, x_idx)
        self.Q = np.zeros((M, M), dtype=float)

    
    def greedy_x_idx(self, y_idx):
        row = self.Q[y_idx]
        maxv = np.max(row)
        candidates = np.flatnonzero(np.isclose(row, maxv))
        return self.rng.choice(candidates)


    def greedy_x(self, y_idx):
        return self.grid[self.greedy_x_idx(y_idx)]

    def td_update(self, y_idx, x_idx, r, y_next_idx):
        td_target = r + self.gamma * np.max(self.Q[y_next_idx])
        td_err = td_target - self.Q[y_idx, x_idx]
        self.Q[y_idx, x_idx] += self.alpha * td_err
        return td_err

    def bayes_update(self, post_y, x_idx, r, post_y_next):
        """
        Bayesian Equation 6-7
          ΔQ(y,x) += α * P(y|H) * (r + γ max_x' E_{y'}[Q(y',x')] - Q(y,x))
        """
        # expected next value under next posterior
        V_next = 0.0
        for y2_idx, py2 in enumerate(post_y_next):
            V_next += py2 * np.max(self.Q[y2_idx])
        td_target = r + self.gamma * V_next

        # update many y states weighted by posterior
        td_err_acc = 0.0
        for y_idx, py in enumerate(post_y):
            td_err = td_target - self.Q[y_idx, x_idx]
            self.Q[y_idx, x_idx] += self.alpha * py * td_err
            td_err_acc += py * abs(td_err)
        return td_err_acc


# estimate nearest grid index
def nearest_grid_idx(grid, v):
    d = np.sum((grid - v[None, :])**2, axis=1)
    return int(np.argmin(d))

# smoothing function
def moving_average(x, win):
    if win <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[win:] - c[:-win]) / win

    pad = np.full(win - 1, y[0])
    return np.concatenate([pad, y])


#  Experiments
def run_hyperq_vs_iga(
    steps=600_000,
    N=25,
    alpha=0.01,
    gamma=0.9,
    mu=0.005,
    bayes_beta=0.995,
    iga_eta=0.05,
    restart_period=1000,
    smooth=5000,
    seed=0
):
    rng = np.random.default_rng(seed)
    grid = simplex_grid_3(N)

    # Learners
    hq_omn = HyperQ(grid, alpha=alpha, gamma=gamma, rng=rng)
    hq_ema = HyperQ(grid, alpha=alpha, gamma=gamma, rng=rng)
    hq_bay = HyperQ(grid, alpha=alpha, gamma=gamma, rng=rng)

    ema = EMAEstimator(mu=mu); ema.reset()
    bay = BayesEstimator(grid=grid, beta=bayes_beta); bay.reset()

    iga = IGAOpponent(eta=iga_eta); iga.reset(rng)

    # Logs
    td_omn = np.zeros(steps)
    td_ema = np.zeros(steps)
    td_bay = np.zeros(steps)

    rew_omn = np.zeros(steps)
    rew_ema = np.zeros(steps)
    rew_bay = np.zeros(steps)

    
    def one_run(estimator_kind):
        rng2 = np.random.default_rng(seed + (0 if estimator_kind=="omn" else 1 if estimator_kind=="ema" else 2))
        grid2 = grid
        hq = HyperQ(grid2, alpha=alpha, gamma=gamma, rng=rng2)
        iga2 = IGAOpponent(eta=iga_eta); iga2.reset(rng2)
        ema2 = EMAEstimator(mu=mu); ema2.reset()
        bay2 = BayesEstimator(grid=grid2, beta=bayes_beta); bay2.reset()

        td = np.zeros(steps)
        rw = np.zeros(steps)

        for t in range(steps):
            if t % restart_period == 0:
                iga2.reset(rng2)
                ema2.reset()
                bay2.reset()

            # choose y estimate
            y_true = iga2.y.copy()
            if estimator_kind == "omn":
                y_est = y_true
                y_idx = nearest_grid_idx(grid2, y_est)
                x_idx = hq.greedy_x_idx(y_idx)
                x = grid2[x_idx]
                # opponent adapts with access to x
                y_true_next = iga2.step(x)
                # observe action from opponent (for omniscient, estimator unused)
                a2 = sample_action(y_true_next, rng2)
                a1 = sample_action(x, rng2)
                r1 = RPS_PAYOFF[a1, a2]
                y_next_idx = nearest_grid_idx(grid2, y_true_next)
                td[t] = abs(hq.td_update(y_idx, x_idx, r1, y_next_idx))
                rw[t] = r1

            elif estimator_kind == "ema":
                y_est = ema2.estimate()
                y_idx = nearest_grid_idx(grid2, y_est)
                x_idx = hq.greedy_x_idx(y_idx)
                x = grid2[x_idx]
                y_true_next = iga2.step(x)
                a2 = sample_action(y_true_next, rng2)
                a1 = sample_action(x, rng2)
                r1 = RPS_PAYOFF[a1, a2]
                # update estimator from observed action
                ema2.update(a2)
                y_est_next = ema2.estimate()
                y_next_idx = nearest_grid_idx(grid2, y_est_next)
                td[t] = abs(hq.td_update(y_idx, x_idx, r1, y_next_idx))
                rw[t] = r1

            else:  # "bay"
                post = bay2.posterior()
                # greedy x via Eq.(7): argmax_x sum_y P(y|H) Q(y,x)
                x_scores = post @ hq.Q  # shape (M,)
                x_idx = int(np.argmax(x_scores))
                x = grid2[x_idx]
                y_true_next = iga2.step(x)
                a2 = sample_action(y_true_next, rng2)
                a1 = sample_action(x, rng2)
                r1 = RPS_PAYOFF[a1, a2]

                # update Bayes with observed action, compute post_next
                bay2.update(a2)
                post_next = bay2.posterior()

                td[t] = hq.bayes_update(post, x_idx, r1, post_next)
                rw[t] = r1

        return td, rw

    td_omn, rew_omn = one_run("omn")
    td_ema, rew_ema = one_run("ema")
    td_bay, rew_bay = one_run("bay")

    return {
        "td": {
            "Omniscient": moving_average(td_omn, smooth),
            "EMA":        moving_average(td_ema, smooth),
            "Bayes":      moving_average(td_bay, smooth),
        },
        "rew": {
            "Omniscient": moving_average(rew_omn, smooth),
            "EMA":        moving_average(rew_ema, smooth),
            "Bayes":      moving_average(rew_bay, smooth),
        }
    }

def run_iga_trajectory(
    steps=40_000,
    N=25,
    alpha=0.01,
    gamma=0.9,
    iga_eta=0.05,
    smooth=0,
    seed=123
):
    rng = np.random.default_rng(seed)
    grid = simplex_grid_3(N)
    hq = HyperQ(grid, alpha=alpha, gamma=gamma, rng=rng)
    iga = IGAOpponent(eta=iga_eta); iga.reset(rng)

    # exploring start with a random y grid point
    y0 = project_simplex(rng.random(3))
    y_idx = nearest_grid_idx(grid, y0)

    rock_prob = np.zeros(steps)
    paper_prob = np.zeros(steps)
    cum_rew = np.zeros(steps)

    total = 0.0
    for t in range(steps):

        # Hyper-Q greedy policy
        x_idx = hq.greedy_x_idx(y_idx)
        x = grid[x_idx]

        # IGA step
        y_true_next = iga.step(x)

        # sample play
        a2 = sample_action(y_true_next, rng)
        a1 = sample_action(x, rng)
        r1 = RPS_PAYOFF[a1, a2]

        # Hyper-Q update with omniscient y 
        y_next_idx = nearest_grid_idx(grid, y_true_next)
        hq.td_update(y_idx, x_idx, r1, y_next_idx)

        y_idx = y_next_idx

        rock_prob[t] = y_true_next[0]
        paper_prob[t] = y_true_next[1]

        total += r1
        cum_rew[t] = total

    # rescale cumulative reward for plotting
    cr = abs(cum_rew)
    if cr.max() > 1e-12:
        cr = cr / cr.max()
    
    
    return moving_average(rock_prob, smooth), moving_average(paper_prob, smooth), cr

def run_hyperq_vs_phc(
    steps=400_000,
    N=25,
    alpha=0.01,
    gamma=0.9,
    mu=0.005,
    bayes_beta=0.995,
    restart_period=1000,
    smooth=5000,
    seed=7
):
    rng = np.random.default_rng(seed)
    grid = simplex_grid_3(N)

    def one_run(kind):
        rng2 = np.random.default_rng(seed + (0 if kind=="omn" else 1 if kind=="ema" else 2))
        hq = HyperQ(grid, alpha=alpha, gamma=gamma, rng=rng2)
        phc = PHCOpponent(alpha_q=0.05, delta=0.002, epsilon=0.0); phc.reset(rng2)
        ema2 = EMAEstimator(mu=mu); ema2.reset()
        bay2 = BayesEstimator(grid=grid, beta=bayes_beta); bay2.reset()

        td = np.zeros(steps)
        rw = np.zeros(steps)

        for t in range(steps):
            if t % restart_period == 0:
                phc.reset(rng2)
                ema2.reset()
                bay2.reset()

            y_true = phc.strategy()

            if kind == "omn":
                y_est = y_true
                y_idx = nearest_grid_idx(grid, y_est)
                x_idx = hq.greedy_x_idx(y_idx)
                x = grid[x_idx]

                a1 = sample_action(x, rng2)
                a2 = phc.act(rng2)
                r1 = RPS_PAYOFF[a1, a2]
                r2 = -r1
                phc.update(a2, r2)

                y_next_idx = nearest_grid_idx(grid, phc.strategy())
                td[t] = abs(hq.td_update(y_idx, x_idx, r1, y_next_idx))
                rw[t] = r1

            elif kind == "ema":
                y_est = ema2.estimate()
                y_idx = nearest_grid_idx(grid, y_est)
                x_idx = hq.greedy_x_idx(y_idx)
                x = grid[x_idx]

                a1 = sample_action(x, rng2)
                a2 = phc.act(rng2)
                r1 = RPS_PAYOFF[a1, a2]
                r2 = -r1
                phc.update(a2, r2)

                ema2.update(a2)
                y_next_idx = nearest_grid_idx(grid, ema2.estimate())
                td[t] = abs(hq.td_update(y_idx, x_idx, r1, y_next_idx))
                rw[t] = r1

            else:  # bayes
                post = bay2.posterior()
                x_scores = post @ hq.Q
                x_idx = int(np.argmax(x_scores))
                x = grid[x_idx]

                a1 = sample_action(x, rng2)
                a2 = phc.act(rng2)
                r1 = RPS_PAYOFF[a1, a2]
                r2 = -r1
                phc.update(a2, r2)

                bay2.update(a2)
                post_next = bay2.posterior()
                td[t] = hq.bayes_update(post, x_idx, r1, post_next)
                rw[t] = r1

        return moving_average(td, smooth), moving_average(rw, smooth)

    td_omn, rw_omn = one_run("omn")
    td_ema, rw_ema = one_run("ema")
    td_bay, rw_bay = one_run("bay")

    return {
        "td": {"Omniscient": td_omn, "EMA": td_ema, "Bayes": td_bay},
        "rew": {"Omniscient": rw_omn, "EMA": rw_ema, "Bayes": rw_bay},
    }

def run_hyperq_selfplay_bellman(
    steps=1_600_000,
    N=25,
    alpha=0.01,
    gamma=0.9,
    bayes_beta=0.995,
    smooth=5000,
    seed=99,
    mode="omn"  # "omn" or "bay"
):
    rng = np.random.default_rng(seed)
    grid1 = simplex_grid_3(N)
    grid2 = simplex_grid_3(N)

    hq1 = HyperQ(grid1, alpha=alpha, gamma=gamma, rng=rng)
    hq2 = HyperQ(grid2, alpha=alpha, gamma=gamma, rng=rng)

    if mode == "bay":
        b1 = BayesEstimator(grid=grid1, beta=bayes_beta); b1.reset()
        b2 = BayesEstimator(grid=grid2, beta=bayes_beta); b2.reset()

    td = np.zeros(steps)

    # initialize
    y1 = project_simplex(rng.random(3))
    y2 = project_simplex(rng.random(3))
    y1_idx = nearest_grid_idx(grid1, y1)
    y2_idx = nearest_grid_idx(grid2, y2)

    for t in range(steps):
        if mode == "omn":
            x1_idx = hq1.greedy_x_idx(y1_idx)
            x2_idx = hq2.greedy_x_idx(y2_idx)

            x1 = grid1[x1_idx]
            x2 = grid2[x2_idx]

            a1 = sample_action(x1, rng)
            a2 = sample_action(x2, rng)

            r1 = RPS_PAYOFF[a1, a2]
            r2 = -r1
            
            # check opponent's next mixed strategy
            y1_next_idx = nearest_grid_idx(grid2, x2)
            y2_next_idx = nearest_grid_idx(grid1, x1)

            e1 = abs(hq1.td_update(y1_idx, x1_idx, r1, y1_next_idx))
            e2 = abs(hq2.td_update(y2_idx, x2_idx, r2, y2_next_idx))
            td[t] = e1

            y1_idx, y2_idx = y1_next_idx, y2_next_idx

        else:
            post1 = b1.posterior()
            post2 = b2.posterior()

            q1_exp = post1 @ hq1.Q 
            maxv1 = np.max(q1_exp)
            candidates1 = np.flatnonzero(np.isclose(q1_exp, maxv1))
            x1_idx =  rng.choice(candidates1)
            q2_exp = post2 @ hq2.Q
            maxv2 = np.max(q2_exp)
            candidates2 = np.flatnonzero(np.isclose(q2_exp, maxv2))
            x2_idx =  rng.choice(candidates2)

            x1 = grid1[x1_idx]
            x2 = grid2[x2_idx]

            a1 = sample_action(x1, rng)
            a2 = sample_action(x2, rng)

            r1 = RPS_PAYOFF[a1, a2]
            r2 = -r1

            # updates Bayes belief with other's policy
            b1.update(a2)
            b2.update(a1)
            post1_next = b1.posterior()
            post2_next = b2.posterior()

            e1 = hq1.bayes_update(post1, x1_idx, r1, post1_next)
            e2 = hq2.bayes_update(post2, x2_idx, r2, post2_next)
            td[t] = e1

    return moving_average(td, smooth)


#  Plotting functions Figure 1–4 

def plot_figure1(results, title_prefix="Hyper-Q vs. IGA", filename="figure1.png"):
    td = results["td"]
    rw = results["rew"]
    T = len(next(iter(td.values())))
    x = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for k, v in td.items():
        axes[0].plot(x, v, label=k)
    axes[0].set_title(f"{title_prefix}: Online Bellman error (smoothed)")
    axes[0].set_xlabel("Time Steps")
    axes[0].set_ylabel("Bellman/TD error")
    axes[0].legend()

    for k, v in rw.items():
        axes[1].plot(x, v, label=k)
    axes[1].set_title(f"{title_prefix}: Avg. reward per time step (smoothed)")
    axes[1].set_xlabel("Time Steps")
    axes[1].set_ylabel("Reward")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(filename)
    #plt.show()

def plot_figure2(rock_prob, paper_prob, cum_rew_scaled):
    T = len(rock_prob)
    x = np.arange(T)

    plt.figure(figsize=(8, 4))
    plt.plot(x, rock_prob, label="IGA_Rock_Prob")
    plt.plot(x, paper_prob, label="IGA_Paper_Prob")

    
    idx = np.arange(0, T, max(1, T // 400))
    plt.scatter(idx, cum_rew_scaled[idx], s=10, label="HyperQ_Reward (rescaled)")

    plt.title("Figure 2 Asymptotic IGA Trajectory ")
    plt.xlabel("Time Steps")
    plt.ylabel("Probability / Rescaled reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure2.png")
    #plt.show()

def plot_figure4(td_omn, td_bay):
    x1 = np.arange(len(td_omn))
    x2 = np.arange(len(td_bay))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x1, td_omn)
    axes[0].set_title("Hyper-Q/Omniscient vs. itself: Online Bellman error (smoothed)")
    axes[0].set_xlabel("Time Steps")
    axes[0].set_ylabel("Bellman/TD error")

    axes[1].plot(x2, td_bay)
    axes[1].set_title("Hyper-Q/Bayes vs. itself: Online Bellman error (smoothed)")
    axes[1].set_xlabel("Time Steps")
    axes[1].set_ylabel("Bellman/TD error")

    plt.tight_layout()
    plt.savefig("figure4.png")


#  Main

if __name__ == "__main__":
    
    # Figure 1 (Hyper-Q vs IGA) 
    res_iga = run_hyperq_vs_iga(
        steps=800_000,
        smooth=5000,
        seed=0
    )
    plot_figure1(res_iga, title_prefix="Hyper-Q vs. IGA", filename="figure1.png")
    

    # Figure 2 trajectory 
    rock, paper, cr = run_iga_trajectory(steps=40_000, seed=777, smooth=100)
    plot_figure2(rock, paper, cr)
    
    # Figure 3 (Hyper-Q vs PHC) 
    res_phc = run_hyperq_vs_phc(
        steps=400_000,
        smooth=5000,
        seed=42
    )
    plot_figure1(res_phc, title_prefix="Hyper-Q vs. PHC", filename="figure3.png")
    
    # Figure 4 (Hyper-Q self-play) 
    td_omn = run_hyperq_selfplay_bellman(
        steps=400_000,    
        smooth=5000,
        seed=123,
        mode="omn"
    )
    td_bay = run_hyperq_selfplay_bellman(
        steps=400_000,    
        smooth=5000,
        seed=123,
        mode="bay"
    )
    plot_figure4(td_omn, td_bay)
