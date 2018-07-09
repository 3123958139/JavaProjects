cons = ({'type': 'eq' , 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))
opts = sco.minimize(min_func_sharpe, noa* [1. / noa, ], method='SLSQP',bounds=bnds, constraints=cons)