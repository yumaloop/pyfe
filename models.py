import cvxopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MarkowitzMinVarianceModel():
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
    def __init__(self, df, window_size, rebalance_freq, r_e=None, r_f=None):
        self.df = self._reset_index(df)
        self.df_chg = self.df.pct_change()
        self.df_chg[:1] = 0.0 # set 0.0 to the first record
        self.df_bt = None
        self.df_bt_r = None
        self.df_bt_x = None
        self.window_size = window_size
        self.rebalance_freq = rebalance_freq
        self.jgb_int = 0.0001 # 0.01% per year (Japanese Government Bond)
        self.r_f = r_f if r_f is not None else self.jgb_int * (1/12) # adjust monthly
        self.r_e = r_e if r_e is not None else r_f
        
    def _reset_index(self, df):
        df = df.copy()
        df['date'] = pd.to_datetime(df.index)
        df = df.set_index('date')
        return df
    
    def get_dfbt_r(self):
        return self.df_bt_r
    
    def get_dfbt_x(self):
        return self.df_bt_x
        
    def backtest(self):
        date_init = self.df.index.values[self.window_size]
        df_bt = pd.DataFrame([[0.0, np.nan]], index=[date_init], columns=['ror', 'std'])
        df_bt_r = pd.DataFrame(columns=list(self.df.columns.values))
        df_bt_x = pd.DataFrame(columns=list(self.df.columns.values))
        for idx, date in enumerate(self.df.index.values):
            if idx >= self.window_size + self.rebalance_freq:
                if (idx - self.window_size) % self.rebalance_freq == 0:
                    # df_chg_train
                    st = idx - self.rebalance_freq - self.window_size
                    ed = idx - self.rebalance_freq
                    df_chg_train = self.df_chg[st:ed]
                    
                    # expected returns per target term
                    if isinstance(self.r_e, pd.core.frame.DataFrame):
                        r_e = self.r_e.iloc[st:ed].values.mean()
                    else:
                        r_e = self.r_e
                    
                    # x_p: min variance portfolio
                    x_p = self.calc_portfolio(df_chg_train, r_e)
                    
                    # df_chg_test
                    st = idx - self.rebalance_freq
                    ed = idx
                    df_chg_test = self.df_chg[st:ed]
                    df_chgcum_test = (1.0 + df_chg_test).cumprod() - 1.0
                                                            
                    # ror_p: rate of return (portfolio)
                    ror_test = df_chgcum_test.iloc[-1].values
                    ror_p = float(np.dot(ror_test, x_p))
                    df_bt_r.loc[date] = ror_test
                    df_bt_x.loc[date] = x_p
                    
                    # std (portfolio)
                    if self.rebalance_freq == 1:
                        std_p = np.nan
                    else:
                        std_test = df_chg_test.std(ddof=True).values
                        std_p = float(np.dot(std_test, np.abs(x_p)))

                    # append
                    df_one = pd.DataFrame([[ror_p, std_p]], index=[date], columns=df_bt.columns)                    
                    df_bt = df_bt.append(df_one)
                    
        # reset index
        self.df_bt = self._reset_index(df_bt)
        self.df_bt_r = self._reset_index(df_bt_r)  
        self.df_bt_x = self._reset_index(df_bt_x)  
        return self.df_bt

    def calc_portfolio(self, df_retchg, r_e):
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
        
        # Adjust params (stop log messages)
        cvxopt.solvers.options['show_progress'] = False # default: True
        cvxopt.solvers.options['maxiters'] = 1000 # default: 100
        
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
    
    def get_yearly_performance(self):
        if self.df_bt is None:
            pass
        else:
            df_yearly = self.df_bt[["ror"]].resample('y').sum()
            df_yearly["std"] = self.df_bt["ror"].resample('y').std().values
            df_yearly["sharpe_ratio"] = df_yearly.apply(lambda d: (d["ror"] - self.r_f) / d["std"], axis=1)
            return df_yearly

    def evaluate_backtest(self, logging=False):   
        if self.df_bt is None:
            pass
        else:
            self.r_mean = self.df_bt["ror"].mean()
            self.r_std = self.df_bt["ror"].std(ddof=True)
            self.sharpe_ratio = (self.r_mean - self.r_f) / self.r_std
            self.net_capgain = (self.df_bt["ror"] + 1.0).cumprod().iloc[-1] - 1.0
            
            self.r_mean_peryear = 12 * self.r_mean
            self.r_std_peryear = np.sqrt(12) * self.r_std
            self.sharpe_ratio_peryear = (self.r_mean_peryear - self.jgb_int) / self.r_std_peryear

            if logging:
                print("Portfolio Performance")
                print("=======================")
                print("Returns per month")
                print("  sharpe ratio     : {:.8f}".format(self.sharpe_ratio))
                print("  mean of returns  : {:.8f}".format(self.r_mean))
                print("  std of returns   : {:.8f}".format(self.r_std))
                print("    risk-free rate : {:.8f}".format(self.r_f))
                print("    capgain ratio  : {:.8f}".format(self.net_capgain))
                print("Returns per year")
                print("  sharpe ratio     : {:.8f}".format(self.sharpe_ratio_peryear))
                print("  mean of returns  : {:.8f}".format(self.r_mean_peryear))
                print("  std of returns   : {:.8f}".format(self.r_std_peryear))
                
            
    def plot_returns(self):
        if self.df_bt is None:
            pass
        else:
            xlabels = [d.strftime('%Y-%m') for idx, d in enumerate(self.df_bt.index) if idx % 12 == 0]
            
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(self.df_bt.index.values, self.df_bt["ror"].values, label="rate of returns")
            ax.plot(self.df_bt.index.values, self.df_bt["ror"].cumsum().values, label="total capital gain ratio")
            ax.legend(loc="upper left")
            ax.set_xticks(xlabels)
            ax.set_xticklabels(xlabels, rotation=40)
            return fig            
        
    def plot_returns_histgram(self):
        if self.df_bt is None:
            pass
        else:
            x = self.df_bt["ror"].values
            r_mean = "{:.4f}".format(x.mean())
            r_std = "{:.4f}".format(x.std())
            
            fig, ax = plt.subplots(figsize=(12,6))
            ax.hist(x, bins=30, alpha=0.75)
            ax.set_title(f"mean={r_mean}, std={r_std}")
            return fig


class SharpeRatioMaxModel():
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
    - r_f: float
        rate of returns of the risk-free asset.
    """
    def __init__(self, df, window_size, rebalance_freq, r_f=None):
        self.df = self._reset_index(df)
        self.df_chg = self.df.pct_change()
        self.df_chg[:1] = 0.0 # set 0.0 to the first record
        self.df_bt = None
        self.df_bt_r = None
        self.df_bt_x = None
        self.window_size = window_size
        self.rebalance_freq = rebalance_freq
        self.jgb_int = 0.0001 # 0.01% per year (Japanese Government Bond)
        self.r_f = r_f if r_f is not None else self.jgb_int * (1/12) # adjust monthly
        
    def _reset_index(self, df):
        df = df.copy()
        df['date'] = pd.to_datetime(df.index)
        df = df.set_index('date')
        return df
    
    def get_dfbt_r(self):
        return self.df_bt_r
    
    def get_dfbt_x(self):
        return self.df_bt_x
        
    def backtest(self):
        date_init = self.df.index.values[self.window_size]
        df_bt = pd.DataFrame([[0.0, np.nan]], index=[date_init], columns=['ror', 'std'])
        df_bt_r = pd.DataFrame(columns=list(self.df.columns.values))
        df_bt_x = pd.DataFrame(columns=list(self.df.columns.values))
        for idx, date in enumerate(self.df.index.values):
            if idx >= self.window_size + self.rebalance_freq:
                if (idx - self.window_size) % self.rebalance_freq == 0:
                    # df_chg_train
                    st = idx - self.rebalance_freq - self.window_size
                    ed = idx - self.rebalance_freq
                    df_chg_train = self.df_chg[st:ed]
                    
                    # x_p: min variance portfolio
                    x_p = self.calc_portfolio(df_chg_train, self.r_f)
                    
                    # df_chg_test
                    st = idx - self.rebalance_freq
                    ed = idx
                    df_chg_test = self.df_chg[st:ed]
                    df_chgcum_test = (1.0 + df_chg_test).cumprod() - 1.0
                                                            
                    # ror_p: rate of return (portfolio)
                    ror_test = df_chgcum_test.iloc[-1].values
                    ror_p = float(np.dot(ror_test, x_p))
                    df_bt_r.loc[date] = ror_test
                    df_bt_x.loc[date] = x_p
                    
                    # std (portfolio)
                    if self.rebalance_freq == 1:
                        std_p = np.nan
                    else:
                        std_test = df_chg_test.std(ddof=True).values
                        std_p = float(np.dot(std_test, np.abs(x_p)))

                    # append
                    df_one = pd.DataFrame([[ror_p, std_p]], index=[date], columns=df_bt.columns)                    
                    df_bt = df_bt.append(df_one)
        
        # reset index
        self.df_bt = self._reset_index(df_bt)
        self.df_bt_r = self._reset_index(df_bt_r)        
        self.df_bt_x = self._reset_index(df_bt_x)
        return self.df_bt

    def calc_portfolio(self, df_retchg, r_f): 
        """ portfolio (sharpe-ratio max model) """
        r = df_retchg.mean().values
        cov = np.array(df_retchg.cov())
        x_opt = self.cvxopt_qp_solver(r, r_f, cov)
        return x_opt
        
    def cvxopt_qp_solver(self, r, r_f, cov):
        """
        CVXOPT QP Solver for Markowitz' Mean-Variance Model
        - See also https://cvxopt.org/userguide/coneprog.html#quadratic-programming
        - See also https://cdn.hackaday.io/files/277521187341568/art-mpt.pdf
        
        r: mean returns of target assets. (vector)
        r_f: rate of returns of the risk-free asset.
        cov: covariance matrix of target assets. (matrix)
        """
        n = len(r)

        # Create Objective matrices
        P = cvxopt.matrix(2.0 * np.array(cov))
        q = cvxopt.matrix(np.zeros((n, 1)))

        # Create constraint matrices
        G = cvxopt.matrix(-np.eye(n))
        h = cvxopt.matrix(np.zeros((n, 1)))
        A = cvxopt.matrix((r - r_f).reshape(1, n))
        b = cvxopt.matrix(1.0)
        
        # stop log messages
        cvxopt.solvers.options['show_progress'] = False
        
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        w_opt = np.squeeze(np.array(sol['x']))
        x_opt = w_opt / np.linalg.norm(w_opt, ord=1)
        return x_opt
    
    def calc_returns(self, df):
        # Rate of returns
        df_retchg = df.pct_change()
        df_retchg[:1] = 0.0 # set 0.0 to the first record
        
        # Cumulative returns)
        df_retcum = (1.0 + df_retchg).cumprod() - 1.0
        df_retcum[:1] = 0.0 # set 0.0 to the first record
        
        return df_retcum, df_retchg
    
    def get_yearly_performance(self):
        if self.df_bt is None:
            pass
        else:
            df_yearly = self.df_bt[["ror"]].resample('y').sum()
            df_yearly["std"] = self.df_bt["ror"].resample('y').std().values
            df_yearly["sharpe_ratio"] = df_yearly.apply(lambda d: (d["ror"] - self.r_f) / d["std"], axis=1)
            return df_yearly
    
    def evaluate_backtest(self, logging=False):   
        if self.df_bt is None:
            pass
        else:
            self.r_mean = self.df_bt["ror"].mean()
            self.r_std = self.df_bt["ror"].std(ddof=True)
            self.sharpe_ratio = (self.r_mean - self.r_f) / self.r_std
            self.net_capgain = (self.df_bt["ror"] + 1.0).cumprod().iloc[-1] - 1.0
            
            self.r_mean_peryear = 12 * self.r_mean
            self.r_std_peryear = np.sqrt(12) * self.r_std
            self.sharpe_ratio_peryear = (self.r_mean_peryear - self.jgb_int) / self.r_std_peryear

            if logging:
                print("Portfolio Performance")
                print("=======================")
                print("Returns per month")
                print("  sharpe ratio     : {:.8f}".format(self.sharpe_ratio))
                print("  mean of returns  : {:.8f}".format(self.r_mean))
                print("  std of returns   : {:.8f}".format(self.r_std))
                print("    risk-free rate : {:.8f}".format(self.r_f))
                print("    capgain ratio  : {:.8f}".format(self.net_capgain))
                print("Returns per year")
                print("  sharpe ratio     : {:.8f}".format(self.sharpe_ratio_peryear))
                print("  mean of returns  : {:.8f}".format(self.r_mean_peryear))
                print("  std of returns   : {:.8f}".format(self.r_std_peryear))
                
            
    def plot_returns(self):
        if self.df_bt is None:
            pass
        else:
            xlabels = [d.strftime('%Y-%m') for idx, d in enumerate(self.df_bt.index) if idx % 12 == 0]
            
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(self.df_bt.index.values, self.df_bt["ror"].values, label="rate of returns")
            ax.plot(self.df_bt.index.values, self.df_bt["ror"].cumsum().values, label="total capital gain ratio")
            ax.legend(loc="upper left")
            ax.set_xticks(xlabels)
            ax.set_xticklabels(xlabels, rotation=40)
            return fig  
        
        
    def plot_returns_histgram(self):
        if self.df_bt is None:
            pass
        else:
            x = self.df_bt["ror"].values
            r_mean = "{:.4f}".format(x.mean())
            r_std = "{:.4f}".format(x.std())
            
            fig, ax = plt.subplots(figsize=(12,6))
            ax.hist(x, bins=30, alpha=0.75)
            ax.set_title(f"mean={r_mean}, std={r_std}")
            return fig