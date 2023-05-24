import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from utils import cov2corr
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.optimize import minimize, Bounds


class EqualWeighted(object):
    """
    - made for daily data
    - assumes daily rebalance
    """

    def __init__(self, eri):
        # TODO Documentation

        ret = eri.pct_change(1).dropna(how='all')
        weights = (~ret.isna() * 1).div(ret.count(axis=1), axis=0)
        ew = ret * weights
        ew = ew.sum(axis=1)
        ew = (1 + ew).cumprod()
        ew = 100 * ew / ew.iloc[0]
        ew = ew.rename('Equal Weighted')

        self.tsweights = weights
        self.weights = weights.iloc[-1].rename('Equal Weighted')
        self.eri = ew


class InverseVol(object):
    """
    - Made for daily data
    - Outputs assumes daily rebalance
    """

    def __init__(self, eri, com=252):

        ret = eri.pct_change(1)
        vols = ret.ewm(com=com, min_periods=com).std() * np.sqrt(252)
        weights = 1 / vols
        weights = weights.div(weights.sum(axis=1), axis=0)

        inv_vol = ret * weights
        inv_vol = inv_vol.dropna(how='all')
        inv_vol = inv_vol.sum(axis=1)
        inv_vol = (1 + inv_vol).cumprod()
        inv_vol = 100 * inv_vol / inv_vol.iloc[0]
        inv_vol = inv_vol.rename('Inverse Vol')

        self.tsweights = weights
        self.weights = weights.iloc[-1].rename('Inverse Vol')
        self.eri = inv_vol


class MinVar(object):
    """
    Implements Minimal Variance Portfolio
    """

    def __init__(self, cov, eri, short_sell=True):
        """
        Combines the assets in 'data' by finding the minimal variance portfolio
        returns an object with the following atributes:
            - 'cov': covariance matrix of the returns
            - 'weights': final weights for each asset
        :param eri: pandas DataFrame where each column is a series of returns
        """

        assert isinstance(eri, pd.DataFrame), "input 'data' must be a pandas DataFrame"

        self.cov = cov

        eq_cons = {'type': 'eq',
                   'fun': lambda w: w.sum() - 1}

        if short_sell:
            bounds = None
        else:
            bounds = Bounds(np.zeros(self.cov.shape[0]), np.ones(self.cov.shape[0]))

        w0 = np.zeros(self.cov.shape[0])
        res = minimize(self._port_var, w0, method='SLSQP', constraints=eq_cons,
                       bounds=bounds, options={'disp': False})

        if not res.success:
            raise ArithmeticError('Convergence Failed')

        self.weights = pd.Series(data=res.x, index=self.cov.columns, name='Min Var')

        ret = eri.pct_change(1)
        minvar = ret * self.weights
        minvar = minvar.dropna()
        minvar = minvar.sum(axis=1)
        minvar = (1 + minvar).cumprod()
        minvar = 100 * minvar / minvar.iloc[0]
        minvar = minvar.rename('Min Var')
        self.eri = minvar

    def _port_var(self, w):
        return w.dot(self.cov).dot(w)


class HRP(object):
    """
    Implements Hierarchical Risk Parity
    """
    # TODO Forçar a escala da matriz de correlação ser de -1 a 1

    def __init__(self, cov, eri, method='single', metric='euclidean'):
        """
        Combines the assets in `data` using HRP
        returns an object with the following attributes:
            - 'cov': covariance matrix of the returns
            - 'corr': correlation matrix of the returns
            - 'sort_ix': list of sorted column names according to cluster
            - 'link': linkage matrix of size (N-1)x4 with structure Y=[{y_m,1  y_m,2  y_m,3  y_m,4}_m=1,N-1].
                      At the i-th iteration, clusters with indices link[i, 0] and link[i, 1] are combined to form
                      cluster n+1. A cluster with an index less than n corresponds to one of the original observations.
                      The distance between clusters link[i, 0] and link[i, 1] is given by link[i, 2]. The fourth value
                      link[i, 3] represents the number of original observations in the newly formed cluster.
            - 'weights': final weights for each asset
        :param data: pandas DataFrame where each column is a series of returns
        :param method: any method available in scipy.cluster.hierarchy.linkage
        :param metric: any metric available in scipy.cluster.hierarchy.linkage
        """

        self.cov = cov
        self.corr, self.vols = cov2corr(self.cov)
        self.method = method
        self.metric = metric

        self.link = self._tree_clustering(self.corr, self.method, self.metric)
        self.sort_ix = self._get_quasi_diag(self.link)
        self.sort_ix = self.corr.index[self.sort_ix].tolist()  # recover labels
        self.sorted_corr = self.corr.loc[self.sort_ix, self.sort_ix]  # reorder correlation matrix
        self.weights = self._get_recursive_bisection(self.cov, self.sort_ix)

        ret = eri.pct_change(1)
        hrp = ret * self.weights
        hrp = hrp.dropna()
        hrp = hrp.sum(axis=1)
        hrp = (1 + hrp).cumprod()
        hrp = 100 * hrp / hrp.iloc[0]
        hrp = hrp.rename('HRP')
        self.eri = hrp

    @staticmethod
    def _tree_clustering(corr, method, metric):
        dist = np.sqrt((1 - corr)/2)
        link = sch.linkage(dist, method, metric)
        return link

    @staticmethod
    def _get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0]*2, 2)  # make space
            df0 = sort_ix[sort_ix >= num_items]  # find clusters
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i+1)
            sort_ix = pd.concat([sort_ix, df0])  # item 2
            sort_ix = sort_ix.sort_index()  # re-sort
            sort_ix.index = range(sort_ix.shape[0])  # re-index
        return sort_ix.tolist()

    def _get_recursive_bisection(self, cov, sort_ix):
        w = pd.Series(1, index=sort_ix, name='HRP')
        c_items = [sort_ix]  # initialize all items in one cluster
        # c_items = sort_ix

        while len(c_items) > 0:

            # bi-section
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]

            for i in range(0, len(c_items), 2):  # parse in pairs
                c_items0 = c_items[i]  # cluster 1
                c_items1 = c_items[i + 1]  # cluster 2
                c_var0 = self._get_cluster_var(cov, c_items0)
                c_var1 = self._get_cluster_var(cov, c_items1)
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                w[c_items0] *= alpha  # weight 1
                w[c_items1] *= 1 - alpha  # weight 2
        return w

    def _get_cluster_var(self, cov, c_items):
        cov_ = cov.loc[c_items, c_items]  # matrix slice
        w_ = self._get_ivp(cov_).reshape(-1, 1)
        c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return c_var

    @staticmethod
    def _get_ivp(cov):
        ivp = 1 / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def plot_corr_matrix(self, save_path=None, show_chart=True, cmap='vlag', linewidth=0, figsize=(10, 10)):
        """
        Plots the correlation matrix
        :param save_path: local directory to save file. If provided, saves a png of the image to the address.
        :param show_chart: If True, shows the chart.
        :param cmap: matplotlib colormap.
        :param linewidth: witdth of the grid lines of the correlation matrix.
        :param figsize: tuple with figsize dimensions.
        """

        sns.clustermap(self.corr, method=self.method, metric=self.metric, cmap=cmap,
                       figsize=figsize, linewidths=linewidth,
                       col_linkage=self.link, row_linkage=self.link)

        if not (save_path is None):
            plt.savefig(save_path)

        if show_chart:
            plt.show()

        plt.close()

    def plot_dendrogram(self, show_chart=True, save_path=None, figsize=(8, 8),
                        threshold=None):
        """
        Plots the dendrogram using scipy's own method.
        :param show_chart: If True, shows the chart.
        :param save_path: local directory to save file.
        :param figsize: tuple with figsize dimensions.
        :param threshold: height of the dendrogram to color the nodes. If None, the colors of the nodes follow scipy's
                           standard behaviour, which cuts the dendrogram on 70% of its height (0.7*max(self.link[:,2]).
        """

        plt.figure(figsize=figsize)
        dn = sch.dendrogram(self.link, orientation='left', labels=self.corr.columns.to_list(),
                            color_threshold=threshold)

        plt.tight_layout()

        if not (save_path is None):
            plt.savefig(save_path)

        if show_chart:
            plt.show()

        plt.close()


class ERC(object):
    """
    Implements Equal Risk Contribution portfolio
    """

    def __init__(self, cov, eri, short_sell=True):
        # TODO Documentation
        self.cov = cov
        self.n_assets = self.cov.shape[0]

        if short_sell:
            bounds = None
        else:
            bounds = np.hstack([np.zeros((self.n_assets, 1)), np.ones((self.n_assets, 1))])

        cons = ({'type': 'ineq',
                 'fun': lambda w: self._port_vol(w)},  # <= 0
                {'type': 'eq',
                 'fun': lambda w: 1 - w.sum()})  # == 0
        w0 = np.zeros(self.n_assets)
        res = minimize(self._dist_to_target, w0, method='SLSQP', constraints=cons, bounds=bounds)
        self.weights = pd.Series(index=eri.columns, data=res.x, name='ERC')
        self.vol = np.sqrt(res.x @ self.cov @ res.x)
        self.marginal_risk = (res.x @ self.cov) / self.vol
        self.risk_contribution = self.marginal_risk * res.x  # These add up to the vol
        self.risk_contribution_ratio = self.risk_contribution / self.vol  # These add up to one

        ret = eri.pct_change(1)
        erc = ret * self.weights
        erc = erc.dropna()
        erc = erc.sum(axis=1)
        erc = (1 + erc).cumprod()
        erc = 100 * erc / erc.iloc[0]
        erc = erc.rename('ERC')
        self.eri = erc

    def _port_vol(self, w):
        return np.sqrt(w.dot(self.cov).dot(w))

    def _risk_contribution(self, w):
        return w * ((w @ self.cov) / (self._port_vol(w)**2))

    def _dist_to_target(self, w):
        return np.abs(self._risk_contribution(w) - np.ones(self.n_assets)/self.n_assets).sum()


class MaxSharpe(object):

    def __init__(self, mu, cov, eri, rf=0, risk_aversion=None, short_sell=True, gross_risk=None):
        """
        Receives informations about the set of assets available for allocation and computes the following
        measures as atributes:
            - cov: Covariance matrix of the assets
            - mu_p: expected return of the optimal risky portfolio (tangency portfolio)
            - sigma_p: risk of the optimal risky portfolio (tangency portfolio)
            - risky_weights: pandas series of the weights of each risky asset on the optimal risk portfolio
            - sharpe_p: expected sharpe ratio of the optimal risky portfolio.
            - mu_mv: expected return of the minimal variance portfolio
            - sigma_mv: risk of the minimal variance portfolio
            - mv_weights: pandas series of the weights of each risky asset on the minimal variance portfolio
            - sharpe_mv: expected sharpe ratio of the minimal variance portfolio.
            - weight_p: weight of the risk porfolio on the investor's portfolio. The remaining 1-weight_p is
                        allocated on the risk-free asset
            - complete_weights: weights of the risky and risk-free assets on the investor's portfolio.
            - mu_c: expected return of the investor's portfolio
            - sigma_c: risk of the investor's portfolio
            - certain_equivalent: the hypothetical risk-free return that would make the investor indiferent
                                  to its risky portfolioo of choice (portfolio C)

        Computations involving the investor's preference use the following utility function:
            U = mu_c - 0.5 * risk_aversion * (sigma_c**2)

        The class also has support for plotting the classical risk-return plane.

        :param mu: pandas series of expected returns where the index contains the names of the assets.
        :param cov: pandas series of risk where the index contains the names of the assets
                    (index must be the same as 'mu')
        :param rf: float, risk-free rate
        :param risk_aversion: coefficient of risk aversion of the investor's utility function.
        :param short_sell: If True, short-selling is allowed. If False, weights on risky-assets are
                           constrained to be between 0 and 1
        :param gross_risk: If a number X is passed, weights cannot go lower than -X or higher than X
        """

        # Assert data indexes and organize
        self._assert_indexes(mu, cov)

        # Save inputs as attributes
        self.mu = mu
        self.cov = cov
        self.rf = rf
        self.risk_aversion = risk_aversion
        self.short_selling = short_sell
        self.gross_risk = gross_risk
        self.n_assets = self._n_assets()

        # Get the optimal risky porfolio
        self.mu_p, self.sigma_p, self.weights, self.sharpe_p = self._get_optimal_risky_portfolio()

        # get the minimal variance portfolio
        self.mu_mv, self.sigma_mv, self.mv_weights, self.sharpe_mv = self._get_minimal_variance_portfolio()

        # Get the investor's portfolio and build the complete set of weights
        if risk_aversion is None:
            self.weight_p = None
            self.complete_weights = None
            self.mu_c = None
            self.sigma_c = None
            self.certain_equivalent = None

        else:
            self.weight_p, self.complete_weights, self.mu_c, self.sigma_c, self.certain_equivalent \
                = self._investor_allocation()

        # Generate the ERI
        ret = eri.pct_change(1)
        mkw = ret * self.weights
        mkw = mkw.dropna()
        mkw = mkw.sum(axis=1)
        mkw = (1 + mkw).cumprod()
        mkw = 100 * mkw / mkw.iloc[0]
        mkw = mkw.rename('Max Sharpe')
        self.eri = mkw

    def plot(self, figsize=None, save_path=None):
        vols = np.sqrt(np.diag(self.cov))

        plt.figure(figsize=figsize)
        # assets
        plt.scatter(vols, self.mu, label='Assets')

        # risk-free
        plt.scatter(0, self.rf, label='Risk-Free')

        # Optimal risky portfolio
        plt.scatter(self.sigma_p, self.mu_p, label='Optimal Risk')

        # Minimal Variance Portfolio
        plt.scatter(self.sigma_mv, self.mu_mv, label='Min Variance')

        # Minimal variance frontier
        if not self.n_assets == 1:
            mu_mv, sigma_mv = self.min_var_frontier()
            plt.plot(sigma_mv, mu_mv, marker=None, color='black', zorder=-1, label='Min Variance Frontier')

        # Capital allocation line
        max_sigma = vols.max() + 0.05
        x_values = [0, max_sigma]
        y_values = [self.rf, self.rf + self.sharpe_p * max_sigma]
        plt.plot(x_values, y_values, marker=None, color='green', zorder=-1, label='Capital Allocation Line')

        # Investor's portfolio
        if self.risk_aversion is not None:
            # Full Portfolio
            plt.scatter(self.sigma_c, self.mu_c, label="Investor's Portfolio", color='purple')

            # Indiference Curve
            max_sigma = self.sigma_c + 0.1
            x_values = np.arange(0, max_sigma, max_sigma / 100)
            y_values = self.certain_equivalent + 0.5 * self.risk_aversion * (x_values ** 2)
            plt.plot(x_values, y_values, marker=None, color='purple', zorder=-1, label='Indiference Curve')

        # legend
        plt.legend(loc='best')

        # adjustments
        plt.xlim((0, vols.max() + 0.1))
        plt.xlabel('Risk')
        plt.ylabel('Return')
        plt.tight_layout()

        # Save as picture
        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def _n_assets(self):
        """
        Makes sure that all inputs have the correct shape and returns the number of assets
        """
        shape_mu = self.mu.shape
        shape_sigma = self.cov.shape

        max_shape = max(shape_mu[0], shape_sigma[0], shape_sigma[1])
        min_shape = min(shape_mu[0], shape_sigma[0], shape_sigma[1])

        if max_shape == min_shape:
            return max_shape
        else:
            raise AssertionError('Mismatching dimensions of inputs')

    def _get_optimal_risky_portfolio(self):

        if self.n_assets == 1:  # one risky asset (analytical)
            mu_p = self.mu.iloc[0]
            sigma_p = self.cov.iloc[0, 0]
            sharpe_p = (mu_p - self.rf) / sigma_p
            weights = pd.Series(data={self.mu.index[0]: 1},
                                name='Max Sharpe')

        else:  # multiple risky assets (optimization)

            # define the objective function (notice the sign change on the return value)
            def sharpe(x):
                return - self._sharpe(x, self.mu.values, self.cov.values, self.rf, self.n_assets)

            # budget constraint
            constraints = ({'type': 'eq',
                           'fun': lambda w: w.sum() - 1})

            # Create bounds for the weights if short-selling is restricted
            if self.short_selling:
                if self.gross_risk is None:
                    bounds = None
                else:
                    bounds = Bounds(-self.gross_risk*np.ones(self.n_assets), self.gross_risk*np.ones(self.n_assets))
            else:
                bounds = Bounds(np.zeros(self.n_assets), np.ones(self.n_assets))

            # initial guess
            w0 = np.zeros(self.n_assets)
            w0[0] = 1

            # Run optimization
            res = minimize(sharpe, w0,
                           method='SLSQP',
                           constraints=constraints,
                           bounds=bounds,
                           options={'ftol': 1e-9, 'disp': False})

            if not res.success:
                raise RuntimeError("Convergence Failed")

            # Compute optimal portfolio parameters
            mu_p = np.sum(res.x * self.mu.values.flatten())
            sigma_p = np.sqrt(res.x @ self.cov @ res.x)
            sharpe_p = - sharpe(res.x)
            weights = pd.Series(index=self.mu.index,
                                data=res.x,
                                name='Max Sharpe')

        return mu_p, sigma_p, weights, sharpe_p

    def _get_minimal_variance_portfolio(self):

        def risk(x):
            return np.sqrt(x @ self.cov @ x)

        # budget constraint
        constraints = ({'type': 'eq',
                        'fun': lambda w: w.sum() - 1})

        # Create bounds for the weights if short-selling is restricted
        if self.short_selling:
            bounds = None
        else:
            bounds = Bounds(np.zeros(self.n_assets), np.ones(self.n_assets))

        # initial guess
        w0 = np.zeros(self.n_assets)
        w0[0] = 1

        # Run optimization
        res = minimize(risk, w0,
                       method='SLSQP',
                       constraints=constraints,
                       bounds=bounds,
                       options={'ftol': 1e-9, 'disp': False})

        if not res.success:
            raise RuntimeError("Convergence Failed")

        # Compute optimal portfolio parameters
        mu_mv = np.sum(res.x * self.mu.values.flatten())
        sigma_mv = np.sqrt(res.x @ self.cov @ res.x)
        sharpe_mv = (mu_mv - self.rf) / sigma_mv
        weights = pd.Series(index=self.mu.index,
                            data=res.x,
                            name='Minimal Variance Weights')

        return mu_mv, sigma_mv, weights, sharpe_mv

    def _investor_allocation(self):
        weight_p = (self.mu_p - self.rf) / (self.risk_aversion * (self.sigma_p**2))
        complete_weights = self.weights * weight_p
        complete_weights.loc['Risk Free'] = 1 - weight_p

        mu_c = weight_p * self.mu_p + (1 - weight_p) * self.rf
        sigma_c = weight_p * self.sigma_p

        ce = self._utility(mu_c, sigma_c, self.risk_aversion)

        return weight_p, complete_weights, mu_c, sigma_c, ce

    def min_var_frontier(self, n_steps=100):

        if self.short_selling:
            E = self.mu.values
            inv_cov = np.linalg.inv(self.cov)

            A = E @ inv_cov @ E
            B = np.ones(self.n_assets) @ inv_cov @ E
            C = np.ones(self.n_assets) @ inv_cov @ np.ones(self.n_assets)

            def min_risk(mu):
                return np.sqrt((C * (mu ** 2) - 2 * B * mu + A)/(A * C - B ** 2))

            min_mu = min(self.mu.min(), self.rf) - 0.01
            max_mu = max(self.mu.max(), self.rf) + 0.05

            mu_range = np.arange(min_mu, max_mu, (max_mu - min_mu) / n_steps)
            sigma_range = np.array(list(map(min_risk, mu_range)))

        else:

            sigma_range = []

            # Objective function
            def risk(x):
                return np.sqrt(x @ self.cov @ x)

            # initial guess
            w0 = np.zeros(self.n_assets)
            w0[0] = 1

            # Values for mu to perform the minimization
            mu_range = np.linspace(self.mu.min(), self.mu.max(), n_steps)

            for mu_step in tqdm(mu_range, 'Finding Mininmal variance frontier'):

                # budget and return constraints
                constraints = ({'type': 'eq',
                                'fun': lambda w: w.sum() - 1},
                               {'type': 'eq',
                                'fun': lambda w: sum(w * self.mu) - mu_step})

                bounds = Bounds(np.zeros(self.n_assets), np.ones(self.n_assets))

                # Run optimization
                res = minimize(risk, w0,
                               method='SLSQP',
                               constraints=constraints,
                               bounds=bounds,
                               options={'ftol': 1e-9, 'disp': False})

                if not res.success:
                    raise RuntimeError("Convergence Failed")

                # Compute optimal portfolio parameters
                sigma_step = np.sqrt(res.x @ self.cov @ res.x)

                sigma_range.append(sigma_step)

            sigma_range = np.array(sigma_range)

        return mu_range, sigma_range

    @staticmethod
    def _sharpe(w, mu, cov, rf, n):
        er = np.sum(w*mu)

        w = np.reshape(w, (n, 1))
        risk = np.sqrt(w.T @ cov @ w)[0][0]

        sharpe = (er - rf) / risk
        return sharpe

    @staticmethod
    def _utility(mu, sigma, risk_aversion):
        return mu - 0.5 * risk_aversion * (sigma ** 2)

    @staticmethod
    def _assert_indexes(mu, cov):
        cond = sorted(mu.index) == sorted(cov.index)
        assert cond, "elements in the input indexes do not match"
