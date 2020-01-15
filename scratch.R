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

## idx <- sample(nrow(sim_dat))
## sim_dat <- sim_dat[idx, ]

N = nrow(sim_dat)
cutoff = 1000

train <- sim_dat[0:cutoff, ]
test <- sim_dat[(cutoff+1):N, ]
## w = rep(1/cutoff, cutoff)

tree <- causalTree(y ~ X1 + X2 + X3 + X4 + X5 + X6,
                   data = train,
                   treatment = sim_dat$treatment,
                   split.Rule = "CT",
                   cv.option = "CT",
                   split.Honest = F,
                   cv.Honest = F,
                   ## HonestSampleSize = 500,
                   split.Bucket = FALSE,
                   xval = 3,
                   minsize = 25)

opcp <- tree$cptable[,1][which.min(tree$cptable[,4])]
t <-prune(tree, opcp)

1 - mean((predict(t, newdata = test) - test$tau)**2) / var(test$tau)

tree <- honest.causalTree(y ~ X1 + X2 + X3 + X4 + X5 + X6,
                   data = train[0:500, ],
                   est_data = train[501:cutoff, ],
                   treatment = sim_dat[0:500, ]$treatment,
                   est_treatment = sim_dat[501:cutoff, ]$treatment,
                   split.Rule = "CT",
                   cv.option = "CT",
                   split.Honest = T,
                   cv.Honest = T,
                   ## HonestSampleSize = 500,
                   split.Bucket = FALSE,
                   xval = 3,
                   minsize = 25)

opcp <- tree$cptable[,1][which.min(tree$cptable[,4])]
t <-prune(tree, opcp)
t
tree
