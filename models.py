import numpy as np
import pandas as pd


class MarkowitzMinVarianceModel():
    
    def __init__(self, df, window_size, rebalance_freq, r_e, r_f=float((1 + 0.0001) ** (1/12) - 1.0)):
        """
        Args:
        =====
        - df: pandas.dataframe
            panel data for target assets for the portfolio. 
                its index must be `numpy.datetime64` type.
                its columns must be time-series data of target assets.
        - window_size: int
            the size of time-window which is used when deriving (or updating) the portfolio.
        - rebalance_freq: int
            rebalance frequency of the portfolio.
        - r_e: float
            min of the return ratio (= capital gain / investment).
        - r_f: float
            rate of returns of the risk-free asset.
        """
        self.df = df
        self.window_size = window_size
        self.rebalance_freq = rebalance_freq
        self.r_e = r_e
        self.r_f = r_f
        
    def backtest(self):
        date_init = df.index.values[self.window_size]
        df_bt = pd.DataFrame([[0.0, 0.0, np.nan, np.nan, np.nan]], index=[date_init], columns=['ror_p', 'ret_p', 'var_p', 'std_p', 'sharperatio_p'])
        for idx, date in enumerate(self.df.index.values):
            if idx >= self.window_size + self.rebalance_freq:
                if idx != 0 and idx % self.rebalance_freq == 0:        
                    # df_train
                    st = idx - self.rebalance_freq - self.window_size
                    ed = idx - self.rebalance_freq
                    df_train = df[st:ed]
                    df_train_retcum, df_train_retchg = self.calc_returns(df_train)
                    
                    # x_p: min variance portfolio
                    x_p = self.mvp(df_train_retchg, self.r_e)
                    
                    # df_test
                    st = idx - self.rebalance_freq
                    ed = idx
                    df_test = df[st:ed]
                    df_test_retcum, df_test_retchg = self.calc_returns(df_test)
                    
                    # ret_p: cum return (portfolio)
                    ret_test = df_test_retcum.iloc[-1].values
                    ret_p = float(np.dot(ret_test, x_p))
                    
                    # ror_p: rate of return (portfolio)
                    ror_test = df_test_retchg.iloc[-1].values
                    ror_p = float(np.dot(ror_test, x_p))
                    
                    # var, std (portfolio)
                    var_test = df_test_retchg.var().values
                    var_p = float(np.dot(var_test, x_p ** 2))
                    std_p = float(np.sqrt(var_p))
                    
                    # sharpe ratio
                    sharperatio_p = self.sharpe_ratio(ret_p, self.r_f, std_p)
                                        
                    # append
                    df_one = pd.DataFrame([[ror_p, ret_p, var_p, std_p, sharperatio_p]], index=[date], columns=df_bt.columns)                    
                    df_bt = df_bt.append(df_one)
        return df_bt

    def mvp(self, df_retchg, r_e):
        r = df_retchg.mean().values
        cov = np.array(df_retchg.cov())
        x_opt = self.cvxopt_qp_solver(r, r_e, cov)
        return x_opt
    
    def sharpe_ratio(self, ret, ret_free, std):
        """
        ret: return of portfolio
        ret_free: return of risk-free asset
        std: standard deviation of portfolio
        """
        return (ret - ret_free) / std
        
    def cvxopt_qp_solver(self, r, r_e, cov):
        """
        CVXOPT QP Solver for Markowitz' Mean-Variance Model
        - See also https://cvxopt.org/userguide/coneprog.html#quadratic-programming
        - See also https://cdn.hackaday.io/files/277521187341568/art-mpt.pdf
        
        r: mean returns of target assets. (vector)
        r_e: min of the return ratio (= capital gain / investment).
        cov: covariance matrix of target assets. (matrix)
        """
        n = len(r)
        r = cvxopt.matrix(r)

        # Create Objective matrices
        P = cvxopt.matrix(2.0 * np.array(cov))
        q = cvxopt.matrix(np.zeros((n, 1)))

        # Create constraint matrices
        G = cvxopt.matrix(np.concatenate((-np.transpose(r), -np.eye(n)), 0))
        h = cvxopt.matrix(np.concatenate((-np.ones((1,1))*r_e, np.zeros((n,1))), 0))
        A = cvxopt.matrix(1.0, (1, n))
        b = cvxopt.matrix(1.0)
        
        # stop log messages
        cvxopt.solvers.options['show_progress'] = False
        
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        x_opt = np.squeeze(np.array(sol['x']))
        return x_opt
    
    def calc_returns(self, df):
        # 収益率(rate of returns)
        df_retchg = df.pct_change()
        df_retchg[:1] = 0.0 # set 0.0 to the first record

        # 累積収益率 (cumulative returns)
        df_retcum = (1.0 + df_retchg).cumprod() - 1.0
        df_retcum[:1] = 0.0 # set 0.0 to the first record
        
        return df_retcum, df_retchg