write_data = function(){
  m = 5
  s = 4
  x = c(rnorm(100, -m, s), rnorm(100, 0, s), rnorm(100, m, s))
  y = c(rnorm(100, -m, s), rnorm(100, m, s), rnorm(100, 0, s))
  mu_x = c(-10, 0, 10)
  mu_y = c(10, 0, -10)
  
  plot(y~x)
  for(i in 1:3)
    points(x = mu_x[i], y = mu_y[i], col = i+1)
  
  unlink(c("x.txt", "y.txt", "mu_x.txt", "mu_y.txt"))
  
  for(i in 1:300){
    write(x[i], "x.txt", append = T)
    write(y[i], "y.txt", append = T)
  }
  
  for(j in 1:3){
    write(mu_x[j], "mu_x.txt", append = T)
    write(mu_y[j], "mu_y.txt", append = T)
  }
}
  
get_dst = function(dst, x, y, mu_x, mu_y, n, k){
  i = 0
  j = 0
  
  for(i in 1:n)
    for(j in 1:k)
      dst[i,j] = (x[i] - mu_x[j])^2 + (y[i] - mu_y[j])^2
  
  .GlobalEnv$dst = dst
}

regroup = function(group, dst, n, k){
  i = 0
  j = 0
  ldst = 0
  
  for(i in 1:n){
    ldst = dst[i,1]
    group[i] = 1
    for(j in 1:k){
      if(dst[i, j] < ldst){
        ldst = dst[i,j]
        group[i] = j
      }
    }
  }
  
  .GlobalEnv$group = group
}  

clear = function(sum_x, sum_y, nx, ny, k){
  j = 0
  for(j in 1:k){
    sum_x[j] = 0
    sum_y[j] = 0
    nx[j] = 0
    ny[j] = 0
  }
  
  .GlobalEnv$sum_x = sum_x
  .GlobalEnv$sum_y = sum_y
  .GlobalEnv$nx = nx
  .GlobalEnv$ny = ny
}

recenter_step1 = function(sum_x, sum_y, nx, ny, x, y, group, n, k){
  i = 0
  j = 0
  
  for(j in 1:k){
    for(i in 1:n){
      if(group[i] == j){
        sum_x[j] = sum_x[j] + x[i]
        sum_y[j] = sum_y[j] + y[i]
        nx[j] = nx[j] + 1
        ny[j] = ny[j] + 1
      }
    }
  }
  
  .GlobalEnv$sum_x = sum_x
  .GlobalEnv$sum_y = sum_y
  .GlobalEnv$nx = nx
  .GlobalEnv$ny = ny
}

recenter_step2 = function(mu_x, mu_y, sum_x, sum_y, nx, ny, k){
  j = 0
  
  for(j in 1:k){
    mu_x[j] = sum_x[j]/nx[j]
    mu_y[j] = sum_y[j]/ny[j]
  }
  
  .GlobalEnv$mu_x = mu_x
  .GlobalEnv$mu_y = mu_y
}

x = scan("x.txt")
y = scan("y.txt")
mu_x = scan("mu_x.txt")
mu_y = scan("mu_y.txt")

n = length(x)
k = 3
nreps = 10

group = rep(0, n)
dst = matrix(0, nrow = n, ncol = k)
sum_x = rep(0, k)
sum_y = rep(0, k)
nx = rep(0, k)
ny = rep(0, k)

plot(y~x, pch = ".", cex = 5, main = paste("K Means Rep", 0))
legend("topright", fill = 1:k, legend = paste("Group", 1:k))
for(i in 1:k)
  points(x = mu_x[i], y = mu_y[i], col = i)
Sys.sleep(2)

for(rep in 1:nreps){
  get_dst(dst, x, y, mu_x, mu_y, n, k)
  regroup(group, dst, n, k)
  clear(sum_x, sum_y, nx, ny, k)
  recenter_step1(sum_x, sum_y, nx, ny, x, y, group, n, k)
  recenter_step2(mu_x, mu_y, sum_x, sum_y, nx, ny, k)

  plot(y~x, col = "white", main = paste("K Means Rep", rep))
  legend("topright", fill = 1:k, legend = paste("Group", 1:k))
  for(i in 1:k){
    points(x = x[group == i], y = y[group == i], col = i, pch = ".", cex = 5)
    points(x = mu_x[i], y = mu_y[i], col = i)
  }

  Sys.sleep(2)  
}