require("ggplot2")
proj.folder <- path.expand("~/Sync/project/2018-leverpush-chloe/data/analysis")  # nolint
fig.folder <- path.expand("~/Sync/project/2018-leverpush-chloe/report/fig") # nolint

# plot decoder between 3 groups
df <- read.csv(paste(proj.folder, "decoder_power.csv", sep = "/"))
df$group <- factor(df$group)
require(MASS)
wilcox.test(df$mutual_info[df$group == 'wt'], df$mutual_info[df$group == 'glt1'])
box <- ggplot(df) + aes(x = group, y = mutual_info, fill = group) + geom_boxplot(outlier.size = F) + geom_jitter(size = 1.9, color = "orange", width = 0.1, show.legend=F)
plot(box)
ggsave(paste(fig.folder, "decoder-comp.svg", sep = "/"), plot = box, width = 9, height = 6)

# paired cno saline comparison
df.diff <- read.csv(paste(proj.folder, "decoder_cno.csv", sep = "/"))
df.cno <- df.diff[df.diff$treat == 'cno',]
df.cno <- df.cno[order(df.cno$case_id), ]
df.saline <- df.diff[df.diff$treat == 'saline',]
df.saline <- df.saline[order(df.saline$case_id), ]
df.cno$order <- 1:(nrow(df.cno))
df.saline$order <- 1:(nrow(df.saline))
df.ordered <- rbind(df.cno, df.saline)
wilcox.test(df.cno$mutual_info, df.saline$mutual_info, paired=T)

box.diff <- ggplot(df.ordered) + aes(x = treat, y = mutual_info, fill = treat) + geom_boxplot(outlier.size = F, width=.5) + geom_jitter(size = 1.9, color = "orange", width = 0.1, show.legend=F) + geom_line(aes(x=treat, y=mutual_info, group=order), color='grey', size=1.)
plot(box.diff)
ggsave(paste(fig.folder, "decoder-cno.svg", sep = "/"), plot = box.diff, width = 9, height = 6)

# slope with neurons
df.single <- read.csv(paste(proj.folder, "single_power.csv", sep = "/"))
df.single <- df.single[df.single$mi > 0, ]
scatter.single <- ggplot(df.single) + aes(x = order, y = mi, color = group) + geom_point() + scale_y_continuous(trans='log2')
plot(scatter.single)
ggsave(paste(fig.folder, "decoder-single-power.svg", sep="/"), plot=scatter.single, width=6, height=6)
df.slope <- read.csv(paste(proj.folder, "single_power_slope.csv", sep = "/"))
scatter.slope <- ggplot(df.slope) + aes(x = group, y = -slope, color = group) + geom_boxplot() + geom_jitter(size=1.9, color='orange', width=0.1, show.legend=F)
plot(scatter.slope)
wilcox.test(df.slope$slope[df.slope$group == 'wt'], df.slope$slope[df.slope$group == 'dredd'])
ggsave(paste(fig.folder, "decoder-single-power-slope.svg", sep="/"), plot=scatter.slope, width=4, height=6)
