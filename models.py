import cvxopt
import numpy as np
import pandas as pd


class MarkowitzMinVarianceModel():
    
    def __init__(self, df, window_size, rebalance_freq, r_e=None, r_f=None):
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
        jgb_int = 0.0001 # 0.01% per year (Japanese Government Bond)
        self.r_f = r_f if r_f is not None else (1 + jgb_int) ** (1/12) - 1.0 # adjust monthly
        self.r_e = r_e if r_e is not None else r_f
        
    def backtest(self):
        date_init = self.df.index.values[self.window_size]
        df_bt = pd.DataFrame([[0.0, np.nan]], index=[date_init], columns=['ror', 'std'])
        for idx, date in enumerate(self.df.index.values):
            if idx >= self.window_size + self.rebalance_freq:
                if idx != 0 and idx % self.rebalance_freq == 0:        
                    # df_train
                    st = idx - self.rebalance_freq - self.window_size
                    ed = idx - self.rebalance_freq
                    df_train = self.df[st:ed]
                    df_train_retcum, df_train_retchg = self.calc_returns(df_train)
                    
                    # x_p: min variance portfolio
                    x_p = self.mvp(df_train_retchg, self.r_e)
                    
                    # df_test
                    st = idx - self.rebalance_freq
                    ed = idx
                    df_test = self.df[st:ed]
                    df_test_retcum, df_test_retchg = self.calc_returns(df_test)
                    
                    # ror_p: rate of return (portfolio)
                    ror_test = df_test_retchg.iloc[-1].values
                    ror_p = float(np.dot(ror_test, x_p))
                    
                    # var, std (portfolio)
                    var_test = df_test_retchg.var().values
                    var_p = float(np.dot(var_test, x_p ** 2))
                    std_p = float(np.sqrt(var_p))
                                                            
                    # append
                    df_one = pd.DataFrame([[ror_p, std_p]], index=[date], columns=df_bt.columns)                    
                    df_bt = df_bt.append(df_one)
        return df_bt

    def mvp(self, df_retchg, r_e):
        r = df_retchg.mean().values
        cov = np.array(df_retchg.cov())
        x_opt = self.cvxopt_qp_solver(r, r_e, cov)
        return x_opt
        
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
        # Rate of returns
        df_retchg = df.pct_change()
        df_retchg[:1] = 0.0 # set 0.0 to the first record
        
        # Cumulative returns)
        df_retcum = (1.0 + df_retchg).cumprod() - 1.0
        df_retcum[:1] = 0.0 # set 0.0 to the first record
        
        return df_retcum, df_retchg
    
    def evaluate_backtest(self, df_backtest, logging=False):            
        self.r_mean = df_backtest["ror"].mean()
        self.r_std = df_backtest["ror"].std() * (len(df_backtest)) / (len(df_backtest)-1)
        self.sharpe_ratio = (self.r_mean - self.r_f) / self.r_std
        self.net_capgain = (df_backtest["ror"] + 1.0).cumprod().iloc[-1] - 1.0
        
        if logging:
            print("Portfolio Performance")
            print("=====================")
            print("mean of returns : {:.8f}".format(self.r_mean))
            print("std of returns  : {:.8f}".format(self.r_std))
            print("risk-free rate  : {:.8f}".format(self.r_f))
            print("sharpe ratio    : {:.8f}".format(self.sharpe_ratio))
            print("capgain ratio   : {:.8f}".format(self.net_capgain))