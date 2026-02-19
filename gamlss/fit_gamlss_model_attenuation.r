# -----------------------------------------------------------------------------
# GAMLSS attenuation model (ST1)
#
# This function fits an age- and covariate-adjusted distributional regression
# model for CT attenuation values using GAMLSS with the ST1 family.
#
# Key idea:
#   - Model the mean (mu) and scale (sigma) as smooth functions of Age using
#     fractional polynomials (fp/bfp), while adjusting for Sex, Manufacturer,
#     optional tube voltage (kVp), and a random Study effect.
#   - Select model complexity by fitting a small set of candidate FP orders
#     (FP_set) and choosing the best candidate by BIC.
#   - Refit a final model using the powers implied by the selected fp() terms
#     (via getSmo) expressed as bfp(Age, powers=...).
#
# Required columns in df:
#   - Volume        : numeric outcome (attenuation / HU; name kept for legacy)
#   - Age           : numeric age (years)
#   - Sex           : factor or character
#   - Manufacturer  : factor or character
#   - Study         : factor or character (used as random(Study))
# Optional columns:
#   - kvp           : numeric tube voltage (required if use_kvp = TRUE)
#
# Returns:
#   A list containing the final fitted GAMLSS model (final_fit_ST1) plus
#   the selected candidate and all candidate fits (for transparency).
#
# -----------------------------------------------------------------------------


fit_gamlss_model_attenuation <- function(df,
                                 family = "ST1",
                                 FP_set = matrix(
                                   c(1, 0, 0,
                                     1, 1, 0,
                                     2, 0, 0,
                                     2, 1, 0,
                                     2, 2, 0,
                                     3, 1, 0,
                                     3, 2, 0,
                                     3, 3, 0),
                                   byrow = TRUE,
                                   ncol  = 3,
                                   dimnames = list(NULL, c("mu","sigma","nu"))
                                 ),
                                 control = gamlss::gamlss.control(c.crit = 1e-2, n.cyc = 90),
                                 use_kvp = FALSE) {

  library(gamlss)                                 

  if (use_kvp && !"kvp" %in% names(df)) {
    stop("use_kvp = TRUE but column 'kvp' is not present in df")
  }
  df <- stats::na.omit(df)

  build_formulas <- function(fp_orders) {
    mu_o    <- fp_orders["mu"]
    sigma_o <- fp_orders["sigma"]
    kvp_term <- if (use_kvp) " + kvp" else ""

    mu_f <- if (mu_o > 0) {
      stats::as.formula(sprintf(
        "Volume ~ 1 + Sex + Manufacturer%s + fp(Age, npoly=%d) + random(Study)",
        kvp_term, mu_o
      ))
    } else {
      stats::as.formula(sprintf(
        "Volume ~ 1 + Sex + Manufacturer%s + random(Study)",
        kvp_term
      ))
    }

    sigma_f <- if (sigma_o > 0) {
      stats::as.formula(sprintf("~ 1 + Sex + fp(Age, npoly=%d) + random(Study)", sigma_o))
    } else {
      ~ 1 + Sex + random(Study)
    }

    nu_f <- ~ 1
    list(mu = mu_f, sigma = sigma_f, nu = nu_f)
  }

  candidate_models <- lapply(seq_len(nrow(FP_set)), function(i) {
    list(family = family, formulas = build_formulas(FP_set[i, ]), meta = as.list(FP_set[i, ]))
  })
  cat("Generated", length(candidate_models), "candidate models.\n")

  fits <- lapply(candidate_models, function(mod) {
    tryCatch({
      fit <- gamlss::gamlss(formula       = mod$formulas$mu,
                            sigma.formula = mod$formulas$sigma,
                            nu.formula    = mod$formulas$nu,
                            family        = mod$family,
                            data          = df,
                            control       = control)
      list(converged = isTRUE(fit$converged),
           AIC       = stats::AIC(fit),
           BIC       = stats::BIC(fit),
           model     = fit,
           meta      = mod)
    }, error = function(e) {
      list(converged = FALSE, AIC = NA_real_, BIC = NA_real_,
           model = NULL, meta = mod, error = e$message)
    })
  })

  conv_idx <- which(vapply(fits, function(x) isTRUE(x$converged), logical(1)))
  if (!length(conv_idx)) stop("No candidate models converged!")

  best_idx_BIC   <- conv_idx[which.min(vapply(fits[conv_idx], `[[`, numeric(1), "BIC"))]
  best_candidate <- fits[[best_idx_BIC]]

  mu_pow    <- gamlss::getSmo(best_candidate$model, what = "mu")$power
  sigma_pow <- gamlss::getSmo(best_candidate$model, what = "sigma")$power

  kvp_term <- if (use_kvp) " + kvp" else ""

  new_mu <- if (length(mu_pow)) {
    stats::as.formula(paste0(
      "Volume ~ 1 + Sex + Manufacturer", kvp_term,
      " + bfp(Age, powers=c(", paste(mu_pow, collapse=","), ")) + random(Study)"
    ))
  } else {
    stats::as.formula(paste0(
      "Volume ~ 1 + Sex + Manufacturer", kvp_term, " + random(Study)"
    ))
  }

  new_sigma <- if (length(sigma_pow)) {
    stats::as.formula(paste0(
      "~ 1 + Sex + bfp(Age, powers=c(", paste(sigma_pow, collapse=","), ")) + random(Study)"
    ))
  } else {
    ~ 1 + Sex + random(Study)
  }

  cat("Refitting final ST1 with BIC-best powers\n")
  final_fit <- gamlss::gamlss(formula       = new_mu,
                              sigma.formula = new_sigma,
                              nu.formula    = ~ 1,
                              family        = family,
                              data          = df,
                              start.from    = best_candidate$model,
                              control       = control)

  final_fit$mydata <- df
  list(final_fit_ST1 = final_fit, best_candidate = best_candidate, candidate_fits = fits)
}
