library(causalTree)
library(dplyr)

tree <- causalTree(y~ x1 + x2 + x3 + x4,
                   data = simulation.1,
                   treatment = simulation.1$treatment,
                   split.Rule = "CT",
                   cv.option = "CT",
                   split.Honest = F,
                   cv.Honest = F,
                   split.Bucket = F,
                   xval = 1, cp = 0, minsize = 20)


ate <- mean(filter(simulation.1, treatment == 1) %>% pull(y)  - filter(simulation.1, treatment == 0) %>% pull(y))

mse <- mean((predict(tree) - (simulation.1$x1 / 2))**2)



sim_dat <- read.csv('sim_dat-2.csv')


N = nrow(sim_dat)
cutoff = round(N/2)

train <- sim_dat[0:cutoff, ]
test <- sim_dat[cutoff:N, ]

tree <- causalTree(y ~ X1 + X2 + X3 + X4 + X5 + X6,
                   data = train,
                   treatment = sim_dat$treatment,
                   split.Rule = "CT",
                   cv.option = "CT",
                   split.Honest = T,
                   cv.Honest = F,
                   split.Bucket = F,
                   xval = 10,
                   split.alpha = 0.1,
                   minsize = 25)



1 - mean((predict(tree, newdata = test) - test$tau)**2) / var(sim_dat$tau)
