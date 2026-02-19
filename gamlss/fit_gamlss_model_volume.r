# -----------------------------------------------------------------------------
# GAMLSS volume model (GG by default): minimal algorithmic reference
#
# This function fits an age- and covariate-adjusted distributional regression
# model for organ volumes using GAMLSS (default family: GG).
#
# Key idea:
#   - Model the location (mu) and scale (sigma) parameters as smooth functions
#     of Age using fractional polynomials (fp/bfp), while adjusting for Sex,
#     Manufacturer, Contrast status, optional tube voltage (kVp), and a random
#     Study effect.
#   - Fit a small grid of candidate models defined by FP_set (orders for fp()
#     terms in mu and sigma; nu kept constant).
#   - Select the best candidate by AIC and by BIC (both are reported), then
#     refit “final” models using the fp() powers extracted via getSmo() and
#     expressed as bfp(Age, powers=...).
#
# Required columns in df:
#   - Volume        : numeric outcome (e.g., organ volume in mL)
#   - Age           : numeric age (years)
#   - Sex           : factor or character
#   - Manufacturer  : factor or character
#   - Contrast      : indicator / factor (contrast-enhanced vs non-contrast)
#   - Study         : factor or character (used as random(Study))
# Optional columns:
#   - kvp           : numeric tube voltage (required if use_kvp = TRUE)
#
# Returns:
#   A list containing:
#     - best_candidate_AIC / best_candidate_BIC (selected candidates)
#     - final_fit_AIC / final_fit_BIC (refit models using bfp() powers)
#     - candidate_fits (all attempted fits with convergence + IC values)

# -----------------------------------------------------------------------------


fit_gamlss_model_volume <- function(df, 
                                    family = "GG",
                                    FP_set = matrix(
                                      c(1, 0, 0,
                                        1, 1, 0,  
                                        2, 0, 0,
                                        2, 1, 0,
                                        2, 2, 0,
                                        3, 1, 0,
                                        3, 2, 0,
                                        3, 3, 0), # candidate 2: mu FP(3), sigma FP(1), nu 0
                                      byrow = TRUE, 
                                      ncol = 3,
                                      dimnames = list(NULL, c("mu", "sigma", "nu"))
                                    ),
                                    use_kvp        = FALSE,                                    
                                    control = gamlss::gamlss.control(n.cyc = 50)) {
  # df: a data.frame containing at least Volume, Sex, Age, Manufacturer, Contrast.
  # family: the GAMLSS family to use.
  # FP_set: a matrix where each row defines candidate FP orders for mu, sigma, and nu.
  # control: control settings for gamlss.

  if (use_kvp && !"kvp" %in% names(df)) {
    stop("use_kvp = TRUE but column 'kvp' is not present in df")
  }
  
  # Omit incomplete cases
  df <- na.omit(df)
  
  library(gamlss)
  
  ## Helper: build formulas given FP orders for mu and sigma.
  build_formulas <- function(fp_orders) {
    mu_order <- fp_orders["mu"]
    sigma_order <- fp_orders["sigma"]
    kvp_term <- if (use_kvp) " + kvp" else ""
    
    # Include nuisance variables Manufacturer and Contrast in the mu model.
    mu_formula <- if (mu_order > 0) {
      sprintf("Volume ~ 1 + Sex + Manufacturer + Contrast%s + fp(Age, npoly=%d) + random(Study)", kvp_term,mu_order)
    } else {
      as.formula(sprintf("Volume ~ 1 + Sex + Manufacturer + Contrast%s + random(Study)", kvp_term))
    }
    # For sigma, we only include Sex (per your previous approach)
    sigma_formula <- if (sigma_order > 0) {
      sprintf("Volume ~ 1 + Sex + fp(Age, npoly=%d) + random(Study)", sigma_order)
    } else {
      "Volume ~ 1 + Sex + random(Study)"
    }
    # nu is modeled as a constant.
    nu_formula <- "Volume ~ 1"
    list(mu = mu_formula, sigma = sigma_formula, nu = nu_formula)
  }
  
  ## Generate candidate models.
  candidate_models <- lapply(seq_len(nrow(FP_set)), function(i) {
    formulas <- build_formulas(FP_set[i, ])
    list(family = family, formulas = formulas, meta = as.list(FP_set[i, ]))
  })
  
  cat("Generated", length(candidate_models), "candidate models.\n")
  
  ## Fit each candidate model.
  fits <- lapply(candidate_models, function(mod) {
    tryCatch({
      fit <- gamlss(formula       = as.formula(mod$formulas$mu),
                    sigma.formula = as.formula(mod$formulas$sigma),
                    nu.formula    = as.formula(mod$formulas$nu),
                    family        = mod$family,
                    data          = df,
                    control       = control)
      list(converged = fit$converged,
           AIC = AIC(fit),
           BIC = BIC(fit),
           model = fit,
           meta = mod)
    }, error = function(e) {
      message("  Candidate failed: ", e$message)
      list(converged = FALSE, AIC = NA_real_, BIC = NA_real_, model = NULL, meta = mod, error = e$message)
    })
  })
  
  ## Keep only candidate models that converged.
  conv_idx <- which(vapply(fits, function(x) x$converged, logical(1)))
  if (length(conv_idx) == 0) stop("No candidate models converged!")
  
  ## Select best candidate by AIC.
  best_idx_AIC <- conv_idx[which.min(sapply(fits[conv_idx], function(x) x$AIC))]
  best_candidate_AIC <- fits[[best_idx_AIC]]
  cat("Best candidate by AIC (index", best_idx_AIC, ") selected.\n")
  cat("  FP orders: mu =", best_candidate_AIC$meta$meta$mu, ", sigma =", best_candidate_AIC$meta$meta$sigma, "\n")
  
  ## Select best candidate by BIC.
  best_idx_BIC <- conv_idx[which.min(sapply(fits[conv_idx], function(x) x$BIC))]
  best_candidate_BIC <- fits[[best_idx_BIC]]
  cat("Best candidate by BIC (index", best_idx_BIC, ") selected.\n")
  cat("  FP orders: mu =", best_candidate_BIC$meta$meta$mu, ", sigma =", best_candidate_BIC$meta$meta$sigma, "\n")
  
  # ---- AIC commented, focusing on BIC ----
  ## Extract smoothing powers using getSmo (assumes this function is available).
  mu_power_AIC    <- getSmo(best_candidate_AIC$model, what = "mu")$power
  sigma_power_AIC <- getSmo(best_candidate_AIC$model, what = "sigma")$power
  nu_power_AIC    <- getSmo(best_candidate_AIC$model, what = "nu")$power
  
  new_mu_AIC <- if (length(mu_power_AIC) > 0) {
    paste0("Volume ~ 1 + Sex + Manufacturer + Contrast + bfp(Age, powers=c(", 
           paste(mu_power_AIC, collapse = ","), ")) + random(Study)")
  } else {
    "Volume ~ 1 + Sex + Manufacturer + Contrast + random(Study)"
  }
new_sigma_AIC <- if (length(sigma_power_AIC) > 0) {
  paste0("Volume ~ 1 + Sex + bfp(Age, powers=c(", paste(sigma_power_AIC, collapse=","), ")) + random(Study)")
} else {
  "Volume ~ 1 + Sex + random(Study)"
}
  new_nu_AIC <- "Volume ~ 1"
  
  cat("Refitting final model based on AIC with formulas:\n")
  cat("  Mu:    ", new_mu_AIC, "\n")
  cat("  Sigma: ", new_sigma_AIC, "\n")
  cat("  Nu:    ", new_nu_AIC, "\n")
  
  final_fit_AIC <- gamlss(formula       = as.formula(new_mu_AIC),
                          sigma.formula = as.formula(new_sigma_AIC),
                          nu.formula    = as.formula(new_nu_AIC),
                          family        = family,
                          data          = df,
                          start.from    = best_candidate_AIC$model)
  
  ## For BIC final model.
  mu_power_BIC    <- getSmo(best_candidate_BIC$model, what = "mu")$power
  sigma_power_BIC <- getSmo(best_candidate_BIC$model, what = "sigma")$power
  nu_power_BIC    <- getSmo(best_candidate_BIC$model, what = "nu")$power
  
  kvp_term <- if (use_kvp) " + kvp" else ""
  new_mu_BIC <- if (length(mu_power_BIC) > 0) {
    paste0("Volume ~ 1 + Sex + Manufacturer + Contrast", kvp_term, "+ bfp(Age, powers=c(", 
           paste(mu_power_BIC, collapse = ","), ")) + random(Study)")
  } else {
    as.formula(paste0("Volume ~ 1 + Sex + Manufacturer", kvp_term, " + Contrast + random(Study)"))
  }
  new_sigma_BIC <- if (length(sigma_power_BIC) > 0) {
    paste0("Volume ~ 1 + Sex + bfp(Age, powers=c(", paste(sigma_power_BIC, collapse = ","), ")) + random(Study)")
  } else {
    "Volume ~ 1 + Sex + random(Study)"
  }
  new_nu_BIC <- "Volume ~ 1"
  
  cat("Refitting final model based on BIC with formulas:\n")
  cat("  Mu:    ", new_mu_BIC, "\n")
  cat("  Sigma: ", new_sigma_BIC, "\n")
  cat("  Nu:    ", new_nu_BIC, "\n")
  
  final_fit_BIC <- gamlss(formula       = as.formula(new_mu_BIC),
                          sigma.formula = as.formula(new_sigma_BIC),
                          nu.formula    = as.formula(new_nu_BIC),
                          family        = family,
                          data          = df,
                          start.from    = best_candidate_BIC$model)
  
  # Attach the complete dataset under a different name so predictions work later.
  final_fit_BIC$mydata <- df
  
  return(list(best_candidate_AIC = best_candidate_AIC,
              final_fit_AIC = final_fit_AIC,
              best_candidate_BIC = best_candidate_BIC,
              final_fit_BIC = final_fit_BIC,
              candidate_fits = fits))
}
