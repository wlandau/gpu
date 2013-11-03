group_out = scan("group_out.txt")

x = scan("x.txt")
y = scan("y.txt")

mu_x_in = scan("mu_x.txt")
mu_y_in = scan("mu_y.txt")

mu_x_out = scan("mu_x_out.txt")
mu_y_out = scan("mu_y_out.txt")

group_in = rep(0, length(x))

for(i in 1:300){
  dist = rep(0, 3)

  for(j in 1:3){
    dist[j] = (x[i] - mu_x_in[j])^2 + (y[i] - mu_y_in[j])^2
  }

  group_in[i] = which.min(dist)
}

pdf("kmeans-before.pdf")
plot(y~x, col = group_in + 1, cex = .5)
for(i in 1:3)
  points(x = mu_x_in[i], y = mu_y_in[i], pch = paste(i), cex = 2)
dev.off()

pdf("kmeans-after.pdf")
plot(y~x, col = group_out + 1, cex = .5)
for(i in 1:3)
  points(x = mu_x_out[i], y = mu_y_out[i], pch = paste(i), cex = 2)
dev.off()