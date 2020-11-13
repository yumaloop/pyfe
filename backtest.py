import numpy as np
import pandas as pd


class MarkowitzMinVarianceModel():
    
    def __init__(self, df, window_size, rebalance_freq, r_e, r_f=0.00001):
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
        df_bt = pd.DataFrame([[0.0, 1.0, np.nan, np.nan]], index=[date_init], columns=['ror_p', 'ret_p', 'var_p', 'sr_p'])
        for idx, date in enumerate(self.df.index.values):
            if idx >= self.window_size + self.rebalance_freq:
                if idx != 0 and idx % self.rebalance_freq == 0:        
                    st_train = idx - self.rebalance_freq - self.window_size
                    ed_train = idx - self.rebalance_freq
                    df_train = df[st_train:ed_train]
                    df_train_retcum, df_train_retchg = self.calc_returns(df_train)
                    x_p = self.mvp(df_train_retchg, self.r_e) # mvp: min var portfolio
                    
                    st_test = idx - self.rebalance_freq
                    ed_test = idx
                    df_test = df[st_test:ed_test]
                    df_test_retcum, df_test_retchg = self.calc_returns(df_test)
                    
                    ret_test = df_test_retcum.iloc[-1].values
                    ret_p = float(np.dot(ret_test, x_p)) # return (portfolio)
                    ror_p = ret_p - 1.0 # rate of return (portfolio)
                    
                    var_test = df_test_retchg.var().values
                    var_p = float(np.dot(var_test, x_p ** 2)) # variance (portfolio)
                    
                    sr_p = self.sharp_ratio(ret_p, self.r_f, var_p)
                                        
                    df_one = pd.DataFrame([[ror_p, ret_p, var_p, sr_p]], index=[date], columns=df_bt.columns)                    
                    df_bt = df_bt.append(df_one)
        return df_bt

    def mvp(self, df_retchg, r_e):
        r = df_retchg.mean().values
        cov = np.array(df_retchg.cov())
        x_opt = self.cvxopt_qp_solver(r, r_e, cov)
        return x_opt
    
    def sharp_ratio(self, ret, ret_free, var):
        """
        ret: return of portfolio
        ret_free: return of risk-free asset
        var: variance of portfolio
        """
        return (ret - ret_free) / np.sqrt(var)
        
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
        P = cvxopt.matrix(2.0 * np.array(cov))
        q = cvxopt.matrix(np.zeros((n, 1)))
        G = cvxopt.matrix(np.concatenate((-np.transpose(r), -np.identity(n)), 0))
        h = cvxopt.matrix(np.concatenate((-np.ones((1,1)) * r_e, np.zeros((n,1))), 0))
        A = cvxopt.matrix(1.0, (1, n))
        b = cvxopt.matrix(1.0)
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        x_opt = np.squeeze(np.array(sol['x']))
        return x_opt
    
    def calc_returns(self, df):
        # 収益率(rate of returns)
        df_retchg = df.pct_change()
        df_retchg[:1] = 0.0 # set 0.0 to the first record

        # 累積収益率 (cumulative returns)
        df_retcum = (1 + df_retchg).cumprod()
        df_retcum[:1] = 1.0 # set 1.0 to the first record
        
        return df_retcum, df_retchg