require("multcomp")
proj.folder <- path.expand("~/Sync/project/2018-leverpush-chloe/data/analysis")  # nolint
fig.folder <- path.expand("~/Sync/project/2018-leverpush-chloe/report/fig")  # nolint
df <- read.csv(paste(proj.folder, "trajectory_shape.csv", sep = "/"))
df$group <- factor(df$group)
df <- df[!(df$case_id %in% c('14032-2', '14032-4', '14032-3', '14032-1')), ]
model <- glht(lm(amplitude ~ group, df))
summary(model)
model <- glht(lm(speed ~ group, df))
summary(model)
# plot
require(ggplot2)
require(ggsignif)
box.amplitude <- ggplot(df, aes(x = group, y = amplitude, fill = group)) + geom_boxplot(outlier.size=F, width=0.5) + geom_jitter(size=1.9, color="orange", width=0.1, show.legend=F)
ggsave(paste(fig.folder, "amplitude.svg", sep="/"), plot=box.amplitude, width=6, height=4)
box.speed <- ggplot(df, aes(x = group, y = speed, fill = group)) + geom_boxplot(outlier.size=F, width=.5) + geom_jitter(size=1.9, color="orange", width=0.1, show.legend=F)
ggsave(paste(fig.folder, "speed.svg", sep="/"), plot=box.speed, width=6, height=4)
box.reliability <- ggplot(df, aes(x = group, y = reliability, fill = group)) + geom_boxplot(outlier.size=F, width=.5) + geom_jitter(size=1.9, color="orange", width=0.1, show.legend=F)
ggsave(paste(fig.folder, "reliability.svg", sep="/"), plot=box.reliability, width=6, height=4)
require(MASS)
# test
wilcox.test(df$speed[df$group == 'glt1'], df$speed[df$group == "wt"])

# hitrate/delay
df <- read.csv(paste(proj.folder, "hitrate_delay.csv", sep = "/"))
df$group <- factor(df$group)
df <- df[!(df$case_id %in% c('14032-2', '14032-4', '14032-3', '14032-1', '18287-1', '18287-2')), ]
box.hitrate <- ggplot(df, aes(x = group, y = hit_rate, fill = group)) + geom_boxplot(outlier.size=F, width=0.5) + geom_jitter(size=1.9, color="orange", width=0.1, show.legend=F)
ggsave(paste(fig.folder, "hitrate.svg", sep="/"), plot=box.hitrate, width=6, height=4)
box.delay <- ggplot(df, aes(x = group, y = delay, fill = group)) + geom_boxplot(outlier.size=F, width=.5) + geom_jitter(size=1.9, color="orange", width=0.1, show.legend=F)
ggsave(paste(fig.folder, "delay.svg", sep="/"), plot=box.delay, width=6, height=4)

wilcox.test(df$hit_rate[df$group == 'dredd'], df$hit_rate[df$group == "wt"])
wilcox.test(df$delay[df$group == 'glt1'], df$delay[df$group == "wt"])
