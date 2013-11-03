# Name: performance.r
# Author: Will Landau (landau@iastate.edu)
# Created: June 2012
#
# This program calculates the runtime of 
# a user-specified function, gpu_function,
# and compares it to that of some cpu
# analog, cpu_function.  
#
# The script creates three plots, each 
# comparing the runtimes of gpu_function 
# to those of cpu_function based on either
# user time, system time, or total scheduled time.


library(multcomp)
library(gputools)


#############
## GLOBALS ##
#############

# functions to compare
cpu_function = function(arg){
  glm(arg[,1] ~ arg[,2:ncol(arg)], family = poisson())
} 

gpu_function = function(arg){
  gpuGlm(arg[,1] ~ arg[,2:ncol(arg)], family = poisson())
}

# global runtime parameters. MUST HAVE length(nrows) == length(ncols) !!!
nrows = 10^(1:4) # nrows of each matrix arg
ncols = 100
sizes = nrows * ncols
xs = log(nrows, base = 10) # plotted on horizontal axis
ys = list() # plotted on vertical axis
xlab = paste("Base 10 log of number of observations (model has ", ncols, "parameters")
title = "glm() vs gpuGlm()"
plot.name = "performance_gpuGlm"
cols = list(cpu = "blue", gpu = "green", outlier.gpu = "black")

# list of arguments
nargs = length(sizes)
args = list()

print("calculating arguments...")
for(i in 1:nargs){
  
  progress = paste("calculating arg ", i, " of ", nargs, sep = "")
  print(progress)

  m = matrix(rnorm(sizes[i]), nrow = nrows[i])
  m[,1] = rpois(n = nrow(m), lambda = 5)
  args[[i]] = m
}

print("done.")

####################
## MAIN FUNCTIONS ##
####################

# iter.time() is a wrapper for one iteration of either 
# gpu_function or cpu_function. The purpose of the wrapper
# is to create the input data, pass the appropriate
# argument (entry arg of the list, args), to gpu_function
# (or cpu_function, if appropriate), and return the run time.

iter.time = function(arg, type = "cpu"){
    if(type == "cpu"){
      ptm <- proc.time()
      cpu_function(arg)
      ptm <- proc.time() - ptm
    } else{
      ptm <- proc.time()
      gpu_function(arg)
      ptm <- proc.time() - ptm
    }

    return(list(user = ptm[1], syst = ptm[2], elap = ptm[3]))
}

# loop.time executes iter.time (i.e., calculates the run time 
# of either gpu_function or cpu_function) for each entry in 
# params (one run per entry in params, each entry defining 
# the magnitude of the computational load on gpu_function or 
# cpu_function).
loop.time = function(args, type = "cpu"){

  user = c()
  syst = c()
  total = c()

  count = 0

  for(arg in args){
    times = iter.time(arg = arg, type = type)    

    user = c(user, times$user)
    syst = c(syst, times$syst)
    total = c(total, times$user + times$syst)

    count = count + 1
    
    progress = paste(type,
                     " iteration ", 
                     count, 
                     " of ",
                     nargs,
                     ": elapsed time = ",
                     times$elap,
                     sep = "")

    print(progress)
  }

  return(list(user = user, syst = syst, total = total))
}


##################
## MAIN ROUTINE ##
##################

# Main routine: actually run gpu_function and cpu_function
# for various data loads and return the run times. Note:
# outlier.gpu.times measures the computational overhead 
# associated with using the gpu for the first time in this
# R script
cpu.times = loop.time(args, type = "cpu")
outlier.gpu.times = loop.time(args[1], type = "gpu") 
gpu.times = loop.time(args, type = "gpu")


#########################################
## FORMAT RUNTIME DATA OF MAIN ROUTINE ##
#########################################

# organize runtime data into a convenient list
times = list(cpu = cpu.times,
             outlier.gpu = outlier.gpu.times,
             gpu = gpu.times)

# format data to plot without confidence
# regions. 
for(time in c("user", "syst", "total")){
  ys[[time]]$outlier.gpu = times$outlier.gpu[[time]]
  for(dev in c("cpu", "gpu")){
    ys[[time]][[dev]] = times[[dev]][[time]]
  }
}

# Format data for plotting: WITH family-wise 
# confidence regions for each run time type
# (time = "user", "syst", or "total") and
# each device (dev = "cpu", "gpu", or 
# "outlier.gpu"). The data, ready for plotting, 
# are available for each device in ys[[time]][[dev]].
#
# for(dev in c("cpu","gpu")){
#  for(time in c("user","syst","total")){
#
#    fit = aov(times[[dev]][[time]] ~ as.factor(sizes) - 1)
#    glht.fit = glht(fit)
#
#    print(glht.fit)
#    
#    if(!all(glht.fit$coef == 0)){
#      famint = confint(glht.fit)
#    } else{
#      zeroes = rep(0,length(glht.fit$coef))
#      famint = list(confint = list(Estimate = zeroes,
#                                        lwr = zeroes,
#                                        upr = zeroes))
#    }
#    
#    ys[[time]][[dev]] = data.frame(famint$confint)
#    ys[[time]]$outlier.gpu = times$outlier.gpu[[time]]
#  }
#}


#######################
## PLOT RUNTIME DATA ##
#######################

# For each kind of run time, make a plot comparing
# the run times of gpu_function to the run times
# of cpu_function.
for(time in c("user", "syst", "total")){
  filename = paste(c(plot.name,"_",time,".pdf"), collapse = "")
  pdf(filename)

  xbounds = c(min(xs), max(xs))
  ybounds = c(min(unlist(ys[[time]])),
              1.3 * max(unlist(ys[[time]])))

  plot(xbounds,
       ybounds,
       pch= ".",
       col="white",
       xlab = xlab,
       ylab = paste(c(time, "scheduled runtime", collapse = " ")),
       main = paste(c(time, "scheduled runtime:", title, collapse = " ")))  

  for(dev in c("cpu", "gpu")){
    points(xs[1], ys[[time]]$outlier.gpu, col=cols$outlier.gpu)
    points(xs, ys[[time]][[dev]], col = cols[[dev]])
    lines(xs, ys[[time]][[dev]], col = cols[[dev]], lty=1)
#    lines(xs, ys[[time]][[dev]]$upr, col = cols[[dev]], lty=1)
  }

  legend("topleft",
         legend = c("mean cpu runtime", 
                    "mean gpu runtime", 
                    "first gpu run (overhead, discarded from conf. region calculations)"),
         col = c(cols$cpu,
                 cols$gpu,
                 "black"),
         pch = c("o"))

  dev.off()
}