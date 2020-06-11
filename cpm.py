import numpy as np
import scipy as sp
from sklearn.linear_model import LogisticRegression

class cpm(object):
    '''
    for prediction of behavioral scores
    '''
    def __init__(self, corr_thresh=0.2):
        '''
        initialize CPM model:
        corr_thresh: retain edges with |correlation to behavior| > corr_thresh
        '''
        self.corr_thresh = corr_thresh

    def _info(self, s):
        print('INFO: %s' %(s))

    def fit(self, X, y):
        '''
        X: upper triangular FC matrices converted to vectors
           (k_subjects x k_edges)
        y: behavioral measure
        '''
        self.k_edges = X.shape[1]

        # mask for edges that exceed corr_thresh
        edge_corr = []
        for edge in range(self.k_edges):
            x = X[:, edge]
            edge_corr.append(np.corrcoef(x, y)[0, 1])

        self.edge_mask = {}
        # edges with positive correlation
        self.edge_mask['pos'] = np.array(edge_corr) > self.corr_thresh
        # edges with positive correlation
        self.edge_mask['neg'] = np.array(edge_corr) < -self.corr_thresh

        '''
        fit linear models
        1. to sum of positive edges only
        2. to sum of negative edges only
        3. to (sum of positive edges) and (sum of negative edges)
        '''
        self.lr_coeff = {}
        # X_glm saves (sum of positive edges) and (sum of negative edges)
        X_glm = np.zeros((X.shape[0], 2))

        for ii, tail in enumerate(self.edge_mask):
            # sum of edges (tail: pos or neg)
            X_sum = X[:, self.edge_mask[tail]].sum(axis=1)
            X_glm[:, ii] = X_sum

            # add column of 1s to X_sum 
            X_sum = np.c_[X_sum, np.ones(X_sum.shape[0])] 

            # linear model, rcond=None uses default thresh for singular values
            self.lr_coeff[tail] = tuple(np.linalg.lstsq(X_sum, y, rcond=None)[0])

        X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
        self.lr_coeff['glm'] = tuple(np.linalg.lstsq(X_glm, y, rcond=None)[0])

        return self.edge_mask, self.lr_coeff 

    def predict(self, X):
        '''
        use lr_coeff to predict y_hat from X
        '''
        y_hat = {}
        # X_glm saves (sum of positive edges) and (sum of negative edges)
        X_glm = np.zeros((X.shape[0], 2))

        # tail: pos, neg
        for ii, tail in enumerate(self.edge_mask):
            # sum of edges (tail: pos or neg)
            X_sum = X[:, self.edge_mask[tail]].sum(axis=1)
            X_glm[:, ii] = X_sum

            # add column of 1s to X_sum 
            X_sum = np.c_[X_sum, np.ones(X_sum.shape[0])]

            # dot product for predictions
            y_hat[tail] = np.dot(X_sum, self.lr_coeff[tail])

        X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
        y_hat['glm'] = np.dot(X_glm, self.lr_coeff['glm'])

        return y_hat

    def score(self, X, y_true):
        '''
        predict y_hat from X
        compare to y_true (true values)
        
        return:
        - R^2 score computed as in: 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score
        best score is 1.0
        - pearsonr
        - spearmanr
        '''
        y_hat = self.predict(X)

        r2 = {}
        for model in y_hat:
            u = ((y_true - y_hat[model]) ** 2).sum()
            v = ((y_true - y_true.mean()) ** 2).sum()
            r2[model] = (1 - u/v)

        pearson = {}
        for model in y_hat:
            pearson[model] = np.corrcoef(y_true, y_hat[model])[0, 1]

        spearman = {}
        for model in y_hat:
            spearman[model] = sp.stats.spearmanr(y_true, y_hat[model])[0]

        return r2, pearson, spearman

    def eval(self, y_hat, y_true):
        '''
        return:
        - R^2 score computed as in: 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score
        best score is 1.0
        - pearsonr
        - spearmanr
        '''
        u = ((y_true - y_hat) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        r2 = (1 - u/v)

        pearson = np.corrcoef(y_true, y_hat)[0, 1]

        spearman = sp.stats.spearmanr(y_true, y_hat)[0]

        return r2, pearson, spearman
