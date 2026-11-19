# Mixed-model version of the treatment analysis (the one in the slides), run on analyzeRCT.py's predictions.csv.
#   Rscript R/glmer.R results/<date-time>/predictions.csv results/<date-time>/glmer prior5 bkt dkt dkt_p irt
# Fits on the events every listed control scored, plus the model without a control ("none"):
#   glmer(outcome ~ treatment * control + (1|user) + (1|problem) + (1|class)), control centered.
# Writes <out_prefix>_terms.csv (odds ratios), _fit.csv (Nakagawa R², AUC, random-effect variances) and
# _pp.csv (percentage-point effect by g-computation, random intercepts kept or set to zero).
# Needs install.packages(c("lme4", "marginaleffects")).
suppressMessages({library(lme4); library(marginaleffects)})
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1]); out <- args[2]; covs <- args[-(1:2)]
d <- d[complete.cases(d[, covs, drop = FALSE]), ]

auc <- function(y, p) { r <- rank(p); n1 <- as.numeric(sum(y == 1)); n0 <- as.numeric(sum(y == 0)); (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
terms <- list(); fits <- list(); pps <- list()
for (cv in c("none", covs)) {
  rhs <- if (cv == "none") "treatment" else { d$cov <- d[[cv]] - mean(d[[cv]]); "treatment * cov" }
  g <- glmer(as.formula(paste("outcome ~", rhs, "+ (1 | user) + (1 | problem) + (1 | class)")),
             data = d, family = binomial, control = glmerControl(optimizer = "bobyqa"))
  s <- coef(summary(g))
  terms[[cv]] <- data.frame(control = cv, term = rownames(s), OR = exp(s[, 1]), lo = exp(s[, 1] - 1.96 * s[, 2]),
                            hi = exp(s[, 1] + 1.96 * s[, 2]), p = s[, 4], row.names = NULL)
  vc <- as.data.frame(VarCorr(g))
  var_f <- var(as.vector(model.matrix(g) %*% fixef(g))); var_r <- sum(vc$vcov)
  fits[[cv]] <- data.frame(control = cv, n = nrow(d),
                           R2_marginal = var_f / (var_f + var_r + pi^2 / 3), R2_conditional = (var_f + var_r) / (var_f + var_r + pi^2 / 3),
                           AUC_fixed = auc(d$outcome, predict(g, re.form = NA)), AUC_full = auc(d$outcome, fitted(g)),
                           tau_user = vc$vcov[vc$grp == "user"], tau_class = vc$vcov[vc$grp == "class"],
                           tau_problem = vc$vcov[vc$grp == "problem"])
  for (rf in c("kept", "zero")) {
    ac <- avg_comparisons(g, variables = list(treatment = c(0, 1)), re.form = if (rf == "kept") NULL else NA)
    pps[[paste(cv, rf)]] <- data.frame(control = cv, random_intercepts = rf, effect_pp = 100 * ac$estimate,
                                       se_pp = 100 * ac$std.error, lo_pp = 100 * ac$conf.low, hi_pp = 100 * ac$conf.high, p = ac$p.value)
  }
  cat("[glmer]", cv, "done\n")
}
write.csv(do.call(rbind, terms), paste0(out, "_terms.csv"), row.names = FALSE)
write.csv(do.call(rbind, fits), paste0(out, "_fit.csv"), row.names = FALSE)
write.csv(do.call(rbind, pps), paste0(out, "_pp.csv"), row.names = FALSE)
