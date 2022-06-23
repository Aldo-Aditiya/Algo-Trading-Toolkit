suppressMessages(library(quantmod))
suppressMessages(library(lattice))
suppressMessages(library(timeSeries))
suppressMessages(library(rugarch))

data <- read.csv(file="log_ret.csv")
arma_order <- as.numeric(scan("arima_model.txt", character()))

spec = ugarchspec(
         variance.model=list(garchOrder=c(1,1)),
         mean.model=list(armaOrder=c(arma_order[1], arma_order[3]), include.mean=T),
         distribution.model="norm")

fit = tryCatch(
    ugarchfit(
     spec, data, solver = 'hybrid'
    ), error=function(e) e, warning=function(w) w
)
    
if(!is(fit, "warning")) {
    print(fit)
    write.csv(residuals(fit, standardize = T), "r-garch-resid.csv")
}