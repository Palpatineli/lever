library("nlme")
library("multcomp")
df <- read.csv("/home/palpatine/Sync/project/2018-leverpush-chloe/data/analysis/classifier_power.csv")
none.data <- subset(df, type == 'none')
mean.data <- subset(df, type == 'mean')
corr.data <- subset(df, type == 'corr')
wt.data <- subset(df, group == 'wt')
dredd.data <- subset(df, group == 'dredd')
glt1.data <- subset(df, group == 'glt1')
##
none.res <- lme(precision ~ group, none.data, random=~1|id)
summary(none.res)
summary(glht(none.res, linfct=mcp(group="Tukey"), test=adjusted(type="bonferroni")))
##
mean.res <- lme(precision ~ group, mean.data, random=~1|id)
summary(mean.res)
summary(glht(mean.res, linfct=mcp(group="Tukey"), test=adjusted(type="bonferroni")))
##
corr.res <- lme(precision ~ group, corr.data, random=~1|id)
summary(corr.res)
summary(glht(corr.res, linfct=mcp(group="Tukey"), test=adjusted(type="bonferroni")))
##
wt.res <- lme(precision ~ type, wt.data, random=~1|id)
summary(wt.res)
summary(glht(wt.res, linfct=mcp(type="Tukey"), test=adjusted(type="bonferroni")))
##
dredd.res <- lme(precision ~ type, dredd.data, random=~1|id)
summary(dredd.res)
summary(glht(dredd.res, linfct=mcp(type="Tukey"), test=adjusted(type="bonferroni")))
##
glt1.res <- lme(precision ~ type, glt1.data, random=~1|id)
summary(glt1.res)
summary(glht(glt1.res, linfct=mcp(type="Tukey"), test=adjusted(type="bonferroni")))
##
glt1.comb <- df[df$type %in% c("none", "corr") & df$group %in% c("wt", "glt1"),]
glt1.comb.res <- lme(precision ~ type * group, glt1.comb, random=~1|id)
summary(glt1.comb.res)
##
dredd.comb <- df[df$type %in% c("none", "corr") & df$group %in% c("wt", "dredd"),]
dredd.comb.res <- lme(precision ~ type * group, dredd.comb, random=~1|id)
summary(dredd.comb.res)
##
means <- aggregate(df$precision, by=list(group=df$group, type=df$type), FUN=mean)
library("reshape2")
dcast(means, group ~ type, value.var='x')
##
mean(df[df$group=='wt' & df$type=='corr', ]$precision)
anova(glt1.comb.res)
anova(dredd.comb.res)
