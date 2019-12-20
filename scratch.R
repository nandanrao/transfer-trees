library(causalTree)


tree <- causalTree(y~ x1 + x2 + x3 + x4,
                   data = simulation.1,
                   treatment = simulation.1$treatment,
                   split.Rule = "CT",
                   cv.option = "CT",
                   split.Honest = F,
                   cv.Honest = F,
                   split.Bucket = F,
                   xval = 1, cp = 0, minsize = 20)
