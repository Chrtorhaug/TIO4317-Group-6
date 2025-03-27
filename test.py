import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.arima.model as sm
import statsmodels.stats.diagnostic as sd
from arch import arch_model
from arch.univariate import EGARCH
from mvgarch.mgarch import DCCGARCH

# S&P 500
spx = pd.read_csv("S&P_500.csv")
spx['Date'] = pd.to_datetime(spx['Date'])
spx["Return"] = spx["Close"].pct_change()
spx = spx.dropna(subset={"Return"})

# Russell 2000
rut = pd.read_csv('Russell_2000.csv')
rut['Date'] = pd.to_datetime(rut['Date'])
rut["Return"] = rut["Close"].pct_change()
rut = rut.dropna(subset={"Return"})

def plot_annualized_volatility(index: pd.DataFrame, model_type: str, symmetry: int):
    model = sm.ARIMA(index['Return'], order=(1, 0, 1)).fit()
    resid = model.resid
    
    p_range = range(1, 5)
    q_range = range(1, 5)
    results = []

    for p in p_range:
        for q in q_range:
            try:
                model = arch_model(resid, vol=model_type, p=p, o=symmetry, q=q, dist='normal', rescale=False)
                fitted_model = model.fit(disp='off')
                results.append({'p': p, 'q': q, 'AIC': fitted_model.aic, 'BIC': fitted_model.bic})
            except Exception as e:
                print(f"Model GARCH({p},{q}) failed: {e}")
                
    results_df = pd.DataFrame(results)
    results_df.sort_values(by='AIC', inplace=True)

    best_aic_model = results_df.loc[results_df['AIC'].idxmin()]
    best_bic_model = results_df.loc[results_df['BIC'].idxmin()]

    print(f"Best model by AIC: GARCH({int(best_aic_model['p'])},{int(best_aic_model['q'])}) with AIC = {best_aic_model['AIC']:.4f}")
    print(f"Best model by BIC: GARCH({int(best_bic_model['p'])},{int(best_bic_model['q'])}) with BIC = {best_bic_model['BIC']:.4f}")

    best_model = arch_model(resid, vol=model_type, p=int(best_aic_model['p']), q=int(best_aic_model['q']), dist='normal', rescale=False)
    fitted = best_model.fit(disp='off')

    # Plot conditional volatility (standard deviation)
    annual_vol = fitted.conditional_volatility*np.sqrt(252)
    mod = "TGARCH" if model_type == "GARCH" and symmetry == 1 else model_type

    plt.figure(figsize=(10, 4))
    plt.plot(index["Date"], annual_vol, label='Conditional Volatility')
    plt.title(f'{mod}({int(best_aic_model["p"])},{int(best_aic_model["q"])}) Conditional Volatility')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    #plt.ylim((0, 0.5))
    plt.legend()
    plt.tight_layout()
    plt.show()