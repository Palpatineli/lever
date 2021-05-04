require("nlme")
require("multcomp")
require("dplyr")
require("ggplot2")
proj.folder <- path.expand(readLines("path.txt")[1])
data.folder <- paste(proj.folder, "data/analysis", sep = "/")
fig.folder <- paste(proj.folder, "report/fig", sep = "/")
df <- read.csv(paste(data.folder, "classifier_power_validated.csv", sep = "/"),
               colClasses = c("factor", "integer", "integer", "factor", "numeric"))
df$case <- paste(df$id, df$session, sep = "+")

##
mean.comb <- df[df$type %in% c("mean", "corr") & df$group %in% c("glt1", "wt"), ]
mean.res <- lme(precision ~ group * type, mean.comb, random = ~1 | case)
glt1.res <- summary(mean.res)
glt1.res$tTable[, "p-value"]["groupwt:typemean"] * 2

mean.comb <- df[df$type %in% c("mean", "corr") & df$group %in% c("dredd", "wt"), ]
mean.res <- lme(precision ~ group * type, mean.comb, random = ~1 | case)
dredd.res <- summary(mean.res)
dredd.res$tTable[, "p-value"]["groupwt:typemean"] * 2

mean.comb <- df[df$type %in% c("mean", "corr") & df$group %in% c("gcamp6f", "wt"), ]
mean.res <- lme(precision ~ group * type, mean.comb, random = ~1 | case)
dredd.res <- summary(mean.res)
dredd.res$tTable[, "p-value"]["groupwt:typemean"] * 2

summary(lme(precision ~ type, df[df$type %in% c("mean", "corr") & df$group %in% c("wt"),], random=~1|case))
dredd1.res <- summary(lme(precision ~ type, df[df$type %in% c("mean", "corr") & df$group %in% c("dredd"),], random=~1|case))
dredd1.res$tTable
summary(lme(precision ~ type, df[df$type %in% c("mean", "corr") & df$group %in% c("glt1"),], random=~1|case))
summary(lme(precision ~ type, df[df$type %in% c("mean", "corr") & df$group %in% c("gcamp6f"),], random=~1|case))

model <- lme(precision ~ group, df[df$type %in% c("mean") & df$group %in% c("wt", "gcamp6f", "dredd", "glt1"),], random=~1|case)
anova(model)
summary(glht(model, mcp(group=c("wt - dredd = 0", "wt - glt1 = 0", "wt - gcamp6f = 0"))))
mean.df <- df[df$type %in% c("mean") & df$group %in% c("wt", "dredd", "glt1"),]
aggregate(mean.df, list(mean.df$group), median)
median(mean.df$precision[mean.df$group == "dredd"])
median(mean.df$precision[mean.df$group == "glt1"])
anova(lme(precision ~ group, df[df$type %in% c("corr") & df$group %in% c("wt", "dredd", "glt1"),], random=~1|case))

##
agg.mean <- aggregate(df$precision, by = list(df$case, df$session, df$type, df$group), FUN = mean, na.rm = T)
agg.mean[order(agg.mean$x), ]
## plot
med <- mean.comb %>% group_by(group, type) %>% summarise(n = median(precision))
box <- (ggplot(mean.comb) + aes(y = precision, fill = interaction(group, type),
                                group = interaction(group, type))
        + geom_boxplot(notch = T, width = .01, alpha = 0.75)
        + geom_hline(yintercept = med$n, linetype = "dashed"))
plot(box)
ggsave(paste(fig.folder, "corr.classifier.svg", sep = "/"), plot = box, width = 6, height = 6)
##
med <- dredd.data %>% group_by(type) %>% summarise(n = median(precision))
box.type <- (ggplot(dredd.data) + aes(x = type, y = precision, fill = type)
             + geom_boxplot(notch = T, width = .5, alpha = 0.75)
             + geom_hline(yintercept = med$n, linetype = "dashed"))
plot(box.type)
ggsave(paste(fig.folder, "dredd.classifier.svg", sep = "/"), plot = box.type, width = 6, height = 6)
##
mean_cl_quantile <- function(x, q = c(0.2, 0.8), na.rm = TRUE){
  dat <- data.frame(y = mean(x, na.rm = na.rm),
                    ymin = quantile(x, probs = q[1], na.rm = na.rm),
                    ymax = quantile(x, probs = q[2], na.rm = na.rm))
  return(dat)
}

df.trace <- read.csv(paste(proj.folder, "cluster_traces.csv", sep = "/"))
cluster.trace <- (ggplot(df.trace, aes(x = time, y = value, group = cluster, color = cluster))
                  + geom_smooth(stat = "summary", fun.data = mean_cl_quantile))
ggsave(paste(fig.folder, "cluster-trace-1.svg", sep = "/"),
       plot = cluster.trace, width = 9, height = 6)
##
