##
proj.folder <- path.expand("~/Sync/project/2018-leverpush-chloe")  # nolint
data.folder <- paste(proj.folder, "data/analysis", sep = "/")
fig.folder <- paste(proj.folder, "report/fig", sep = "/")

##
df <- read.csv(paste(data.folder, "encoding_mean.csv", sep = "/"))
df$group <- factor(df$group)

##
df <- read.csv(paste(data.folder, "encoding.csv", sep = "/"))
var.names <- colnames(df)[2: 10]
df$group <- factor(df$group)
df$id <- factor(df$id)

for (colname in var.names) {
    a <- df[df$group == "glt1", colname]
    b <- df[df$group == "wt", colname]
    a[a < 0] <- 0
    b[b < 0] <- 0
    res <- wilcox.test(a, b)
    print(c(colname, " p: ", res$p.value, " greater than wt: ", (median(a) > median(b))))
}

##
df <- read.csv(paste(data.folder, "encoding.csv", sep = "/"))
var.names <- colnames(df)[2: 10]
df$group <- factor(df$group)
df$id <- factor(df$id)
df <- df[df$delay_length >= 0, ]
require(ggplot2)
dist.delay <- ggplot(df, aes(x = delay_length, fill = group))
dist.delay <- dist.delay + geom_density(alpha = .5)
dist.delay <- dist.delay + coord_cartesian(ylim = c(0, 5))
ggsave(paste(fig.folder, "encoding-delay-length.svg", sep = "/"),
       plot = dist.delay, width = 6, height = 4)

##
require("nlme")
require("multcomp")
for (colname in var.names) {
    formula <- as.formula(paste(colname, "group", sep = "~"))
    res <- lme(formula, df, random = ~1 | id)
    summary(glht(res, linfct = mcp(group = "Tukey")))
    comp.res <- glht(res, linfct = mcp(group = "Tukey"))
    print(paste(colname, ": "))
    print(paste(summary(comp.res)$test$pvalues, collapse = ", "))
}
##
