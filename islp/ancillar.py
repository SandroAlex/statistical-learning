# Built-in imports
from functools import partial
from typing import Tuple, List

# Data science imports
import numpy as np
import pandas as pd

# Linear regression from statsmodels
from statsmodels.api import OLS


def nCp(
        sigma2: float, 
        estimator: OLS,
        X: pd.DataFrame, 
        Y: pd.DataFrame
    ) -> float:
    """
    Negative Cp statistic.
    """
    
    n, p = X.shape
    Yhat = estimator.predict(X)
    RSS = np.sum((Y - Yhat)**2)
    
    return -(RSS + 2 * p * sigma2) / n


def adjust_linear_model(
        dfX: pd.DataFrame, 
        dfY: pd.DataFrame
    ) -> Tuple[OLS, float, float, float, float]:
    """
    Adjust a linear model with all predictors.
    """
    
    linear_model = OLS(endog=dfY, exog=dfX).fit()

    return linear_model, linear_model.ssr, linear_model.rsquared, linear_model.bic, linear_model.aic


def forward_stepwise_selection(
        dfX: pd.DataFrame, 
        dfY: pd.DataFrame,
        sigma2: float,
        verbose: bool = False
    ) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float], List[OLS]]:
    """
    Forward stepwise selection.
    """
    
    if verbose:
        print(">>> Starting forward stepwise selection...")

    neg_Cp = partial(nCp , sigma2)

    # Initially all predictor are available.
    all_features: List[str] = dfX.columns.to_list()
    available_features: List[str] = dfX.columns.to_list()
    used_features: List[str] = []

    # Initialize some metrics to be used as a function of the number of used features.
    n_features: List[int] = []
    rss_list: List[float] = []
    r2_list: List[float] = []
    bic_list: List[float] = []
    aic_list: List[float] = []
    cp_list: List[float] = []
    best_models: List[OLS] = []

    # Initialize best Cp as very low.
    best_cp: float = -np.inf

    while len(used_features) < len(all_features):

        # Loop over all predictors.
        for feature in available_features:
            
            # If feature not already used.
            if feature not in used_features:

                # Candidates to test.
                tested_features: List[str] = used_features + [feature]

                if verbose:
                    print(f">>> Testing features: {tested_features}")

                # Fit model with current feature.
                model, rss, r2, bic, aic = adjust_linear_model(dfX[tested_features], dfY)

                # Calculate cp.
                cp: float = neg_Cp(model, dfX[tested_features], dfY["Y"])

                if verbose:
                    print(f"\t* metrics: CP={cp:.2f} (bestCP={best_cp:.2f}), RSS={rss:.2f}, R2={r2:.2f}, BIC={bic:.2f}, AIC={aic:.2f}")

                # Verify if the model is better according to Cp.
                if cp > best_cp:

                    # Update best metrics.
                    best_cp: float = cp
                    best_rss: float = rss
                    best_r2: float = r2
                    best_bic: float = bic
                    best_aic: float = aic

                    # Update best feature.
                    best_feature: str = feature

                    # Best model.
                    best_model: OLS = model

        # Update lists.
        used_features.append(best_feature)
        n_features.append(len(used_features))
        rss_list.append(best_rss)
        r2_list.append(best_r2)
        bic_list.append(best_bic)
        aic_list.append(best_aic)
        cp_list.append(best_cp)
        available_features.remove(best_feature)
        best_models.append(best_model)

        if verbose:
            print(f"\t* Diagnostic: used features: {used_features}, Cp: {best_cp:.2f}, RSS: {best_rss:.2f}, R2: {best_r2:.2f}")

        # Reset best Cp for next round.
        best_cp = -np.inf

    if verbose:
        print("\n>>> Done!")    

    return n_features, rss_list, r2_list, bic_list, aic_list, cp_list, best_models


def backward_stepwise_selection(
        dfX: pd.DataFrame, 
        dfY: pd.DataFrame,
        verbose: bool = False
    ) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float], List[OLS]]:
    """
    Backward stepwise selection.
    """
    
    # Skeleton for backward stepwise selection.
    all_features: List[str] = dfX.columns.to_list()
    tested_features: List[str] = all_features.copy()
    candidates: List[str] = all_features.copy()

    # Initialize some metrics to be used as a function of the number of used features.
    n_features: List[int] = []
    rss_list: List[float] = []
    r2_list: List[float] = []
    bic_list: List[float] = []
    aic_list: List[float] = []
    cp_list: List[float] = []
    best_models: List[OLS] = []

    # Adjust linear model with all predictors.
    model, rss, r2, bic, aic = adjust_linear_model(dfX, dfY)

    # Calculate cp.
    cp: float = neg_Cp(model, dfX[tested_features], dfY["Y"])    

    n_features.append(len(all_features))
    rss_list.append(rss)
    r2_list.append(r2)
    bic_list.append(bic)
    aic_list.append(aic)
    cp_list.append(cp)
    best_models.append(model)

    # Initialize best Cp as very low.
    best_cp: float = -np.inf

    while len(candidates) > 1:

        if verbose:
            print(f">>> Candidates to remove: {candidates}")

        # Loop over candidates to be removed.
        for candidate in all_features:

            # Is this a valid candidate?
            if candidate in candidates:

                # Features to be tested.
                tested_features = natsorted(list(set(candidates) - set([candidate])))

                if verbose:
                    print(f"\t* Exclude {candidate}:")

                # Adjust model and calculate metrics.
                model, rss, r2, bic, aic = adjust_linear_model(dfX[tested_features], dfY)

                # Calculate cp.
                cp: float = neg_Cp(model, dfX[tested_features], dfY["Y"])

                if verbose:
                    print(f"\t\t* metrics: CP={cp:.2f} (bestCP={best_cp:.2f}), RSS={rss:.2f}, R2={r2:.2f}, BIC={bic:.2f}, AIC={aic:.2f}")

                # Verify if the model is better according to Cp.
                if cp > best_cp:

                    # Update best metrics.
                    best_cp: float = cp
                    best_rss: float = rss
                    best_r2: float = r2
                    best_bic: float = bic
                    best_aic: float = aic

                    # Update best feature.
                    removed_feature: str = candidate
                    used_features: List[str] = tested_features

                    # Best model.
                    best_model: OLS = model
                
        # Select feature to be removed.
        if verbose:
            print(f"\t* Removed feature: {removed_feature}")
        candidates.remove(removed_feature)    

        # Update lists.
        n_features.append(len(used_features))
        rss_list.append(best_rss)
        r2_list.append(best_r2)
        bic_list.append(best_bic)
        aic_list.append(best_aic)
        cp_list.append(best_cp)
        best_models.append(best_model)

        if verbose:
            print(f"\t* Diagnostic:")
            print(f"\t\t* used features: {used_features}")
            print(f"\t\t* Cp: {best_cp:.2f}, RSS: {best_rss:.2f}, R2: {best_r2:.2f}")

        # Reset best Cp for next round.
        best_cp = -np.inf    

    return n_features, rss_list, r2_list, bic_list, aic_list, cp_list, best_models
