library(coda)

r1 = read.csv("results1.csv")
r2 = read.csv("results2.csv")

a = mcmc.list(mcmc(r1$a[2001:10000]), mcmc(r2$a[2001:10000]))
b = mcmc.list(mcmc(r1$b[2001:10000]), mcmc(r2$b[2001:10000]))

gelman.diag(a, autoburnin = F)
gelman.diag(b, autoburnin = F)

pdf("mcmc-trace-alpha.pdf")
plot(a, main = "Alpha")
dev.off()

pdf("mcmc-trace-beta.pdf")
plot(b, main = "Beta")
dev.off()