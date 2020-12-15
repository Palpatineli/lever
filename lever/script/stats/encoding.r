##
proj.folder <- path.expand(readLines("path.txt")[1])
data.folder <- paste(proj.folder, "data/analysis", sep = "/")
fig.folder <- paste(proj.folder, "report/fig", sep = "/")

##
df <- read.csv(paste(data.folder, "encoding_mean.csv", sep = "/"))
df$group <- factor(df$group)

##
df <- read.csv(paste(data.folder, "encoding_minimal.csv", sep = "/"))
var.names <- colnames(df)[2: 8]
# df$group <- factor(df$group)
# df$id <- factor(df$id)
df$id <- with(df, interaction(factor(case_id), factor(session_id)))

options(warn = -1)
all <- df[, c("all", "group")]
all$all[all$all < 0] <- 0
xa <- all$all[all$group == "wt"]
ya <- all$all[all$group == "glt1"]
za <- all$all[all$group == "dredd"]
coef <- 3
for (colname in var.names) {
    x <- df[df$group == "wt", colname]
    y <- df[df$group == "glt1", colname]
    z <- df[df$group == "dredd", colname]
    y[y < 0] <- 0
    x[x < 0] <- 0
    z[z < 0] <- 0
    print("")
    print(colname)
    print(paste0(c("wt", "glt1", "dreadd")[order(c(median(x), median(y), median(z)))], collapse = " < "))
    print(paste0("wt-glt1:", wilcox.test(x, y)$p.value * coef, "    wt-glt1 ks:", ks.test(x, y)$p.value * coef))
    print(paste0("glt1-dredd:", wilcox.test(y, z)$p.value * coef, "    glt1-dredd ks:", ks.test(y, z)$p.value * coef))
    print(paste0("dredd-wt:", wilcox.test(z, x)$p.value * coef, "    dredd-wt ks:", ks.test(z, x)$p.value * coef))
    print(paste0(c("wt", "glt1", "dreadd")[order(c(median(x / xa), median(y / ya), median(z / za)))], collapse = " < "))
    print(paste0("wt-glt1 / all:", wilcox.test(x / xa, y / ya)$p.value * coef, "    wt-glt1 ks:", ks.test(x / xa, y / ya)$p.value * coef))
    print(paste0("glt1-dredd / all:", wilcox.test(y / ya, z / za)$p.value * coef, "    glt1-dredd ks:", ks.test(y / ya, z / za)$p.value * coef))
    print(paste0("dredd-wt / all:", wilcox.test(z / za, x / xa)$p.value * coef, "    dredd-wt ks:", ks.test(z / za, x / xa)$p.value * coef))
}
options(warn = 0)

wilcox.test(df$all[df$group=="wt"], df$all[df$group=="glt1"])
wilcox.test(df$all[df$group=="wt"], df$all[df$group=="dredd"])
## ARTool two way
require("ARTool")
require("tidyr")

art_2by2 <- function(df, feature, group){
    feature_wide <- df[,c("X", feature, "all", "group")]
    feature_long <- gather(feature_wide, time, r2, c(feature, "all"), factor_key=TRUE)
    feature_long$r2[feature_long$r2 < 0] <- 0
    feature_long$X <- factor(feature_long$X)
    feature_long <- feature_long[feature_long$group %in% c("wt", group), ]
    model <- art(r2 ~ group * time, data=feature_long)
    return(model)
}

wilcox <- function(df, feature) {
    res <- list()
    for (group_str in c("wt", "dredd", "glt1")) {
        temp <- df[df$group == group_str, feature]
        temp[temp < 0] <- 0
        res[group_str] <- temp
    }
    wt <- df[df$group == "wt", feature]
    wt[wt < 0] <- 0
    median(wt)
    dredd <- df[df$group 
    return(list(wilcox.test(, df[df$group == "dredd", feature]),
                wilcox.test(df[df$group == "wt", feature], df[df$group == "glt1", feature])))
}
##
feature <- "reward"
median(df[df$group == "wt", feature])
median(df[df$group == "dredd", feature])
##
glt1.res <- anova(art_2by2(df, "delay", "glt1"))
glt1.res[, "Pr(>F)"]
dredd.res <- anova(art_2by2(df, "delay", "dredd"))
dredd.res[, "Pr(>F)"]
wilcox(df, "start")
head(df)
wilcox(df, "reward")
wilcox(df, "isMoving")
wilcox(df, "trajectory")
wilcox(df, "speed")
##
##
df <- read.csv(paste(data.folder, "encoding.csv", sep = "/"))
df$group <- factor(df$group)
df$id <- factor(df$id)
var.names <- colnames(df)[2: 10]
for (colname in var.names) {
    df[df[, colname] < 0, colname] <- 0
}
require(ggplot2)
dist.delay <- ggplot(df, aes(x = delay_length, fill = group))
dist.delay <- dist.delay + geom_density(alpha = .5)
dist.delay <- dist.delay + coord_cartesian(ylim = c(0, 4), xlim = c(0.025, 0.28))
ggsave(paste(fig.folder, "encoding-delay-length.svg", sep = "/"),
       plot = dist.delay, width = 6, height = 3)

df$speed[df$speed < 0] <- 0
dist.speed <- ggplot(df, aes(x = speed, fill = group))
dist.speed <- dist.speed + geom_density(alpha = .5)
dist.speed <- dist.speed + coord_cartesian(ylim = c(0, 25), xlim = c(0.0, 0.15))
ggsave(paste(fig.folder, "encoding-delay-speed.svg", sep = "/"),
       plot = dist.speed, width = 6, height = 3)


##
require("nlme")
require("multcomp")
var.names <- colnames(df)[2: 10]
for (colname in var.names) {
    formula <- as.formula(paste(colname, "group", sep = "~"))
    res <- lme(formula, df, random = ~1 | id)
    summary(glht(res, linfct = mcp(group = "Tukey")))
    comp.res <- glht(res, linfct = mcp(group = "Tukey"))
    print(paste(colname, ": "))
    print(paste(summary(comp.res)$test$pvalues, collapse = ", "))
}
##
