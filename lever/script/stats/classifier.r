library("nlme")
library("multcomp")
proj.folder <- path.expand("~/Sync/project/2018-leverpush-chloe/data/analysis")  # nolint
fig.folder <- path.expand("~/Sync/project/2018-leverpush-chloe/report/fig")  # nolint
df <- read.csv(paste(proj.folder, "classifier_power.csv", sep = "/"))
none.data <- subset(df, type == "none")
mean.data <- subset(df, type == "mean")
corr.data <- subset(df, type == "corr")
wt.data <- subset(df, group == "wt")
dredd.data <- subset(df, group == "dredd")
glt1.data <- subset(df, group == "glt1")
##
none.res <- lme(precision ~ group, none.data, random = ~1 | id)
summary(none.res)
summary(glht(none.res, linfct = mcp(group = "Tukey"), test = adjusted(type = "bonferroni")))
##
mean.res <- lme(precision ~ group, mean.data, random = ~1 | id)
summary(mean.res)
summary(glht(mean.res, linfct = mcp(group = "Tukey"), test = adjusted(type = "bonferroni")))
##
corr.res <- lme(precision ~ group, corr.data, random = ~1 | id)
summary(corr.res)
summary(glht(corr.res, linfct = mcp(group = "Tukey"), test = adjusted(type = "bonferroni")))
t.test(corr.data[corr.data$group == "dredd", ]$precision, corr.data[corr.data$group == "wt", ]$precision)
t.test(corr.data[corr.data$group == "glt1", ]$precision, corr.data[corr.data$group == "wt", ]$precision)
##
wt.res <- lme(precision ~ type, wt.data, random = ~1 | id)
summary(wt.res)
summary(glht(wt.res, linfct = mcp(type = "Tukey"), test = adjusted(type = "bonferroni")))
##
dredd.res <- lme(precision ~ type, dredd.data, random = ~1 | id)
summary(dredd.res)
summary(glht(dredd.res, linfct = mcp(type = "Tukey"), test = adjusted(type = "bonferroni")))
##
glt1.res <- lme(precision ~ type, glt1.data, random = ~1 | id)
summary(glt1.res)
summary(glht(glt1.res, linfct = mcp(type = "Tukey"), test = adjusted(type = "bonferroni")))
##
glt1.comb <- df[df$type %in% c("none", "corr") & df$group %in% c("wt", "glt1"), ]
glt1.comb.result <- lme(precision ~ type * group, glt1.comb, random = ~1 | id)
summary(glt1.comb.result)
##
dredd.comb <- df[df$type %in% c("none", "corr") & df$group %in% c("wt", "dredd"), ]
dredd.comb.res <- lme(precision ~ type * group, dredd.comb, random = ~1 | id)
summary(dredd.comb.res)
##
means <- aggregate(df$precision, by = list(group = df$group, type = df$type), FUN = mean)
library("reshape2")
dcast(means, group ~ type, value.var = "x")
##
mean(df[df$group == "wt" & df$type == "corr", ]$precision)
anova(glt1.comb.res)
anova(dredd.comb.res)
## plot
require(ggplot2)
require(dplyr)
med <- corr.data %>% group_by(group) %>% summarise(n = median(precision))
box <- ggplot(corr.data) + aes(x = group, y = precision, fill = group)
    + geom_boxplot(notch = T, width = .5, alpha = 0.75)
    + geom_hline(yintercept = med$n, linetype = "dashed")
plot(box)
ggsave(paste(fig.folder, "corr.classifier.svg", sep = "/"), plot = box, width = 6, height = 6)
##
med <- dredd.data %>% group_by(type) %>% summarise(n = median(precision))
box.type <- ggplot(dredd.data) + aes(x = type, y = precision, fill = type) + geom_boxplot(notch = T, width = .5, alpha = 0.75) + geom_hline(yintercept = med$n, linetype = "dashed")
plot(box.type)
ggsave(paste(fig.folder, "dredd.classifier.svg", sep = "/"), plot = box.type, width = 6, height = 6)
##
require(ggplot2)
mean_cl_quantile <- function(x, q = c(0.2, 0.8), na.rm = TRUE){
  dat <- data.frame(y = mean(x, na.rm = na.rm),
                    ymin = quantile(x, probs = q[1], na.rm = na.rm),
                    ymax = quantile(x, probs = q[2], na.rm = na.rm))
  return(dat)
}

df.trace <- read.csv(paste(proj.folder, "cluster_traces.csv", sep="/"))
cluster.trace <- ggplot(df.trace, aes(x = time, y = value, group=cluster, color=cluster)) + geom_smooth(stat = 'summary', fun.data = mean_cl_quantile)
ggsave(paste(fig.folder, "cluster-trace-1.svg", sep = "/"), plot = cluster.trace, width = 9, height = 6)
##
