print('hola')

png(filename="vectores.png")
plot(c(0,10),c(0,10))
points(c(2,8), c(2,2), type='l')
points(c(2,2), c(2,8), type='l')
arrows(2,2,5,4, lwd=4)
text(3.5,3.5,'a', cex=2)
arrows(2,2,5,2, lwd=4, col='red')
arrows(2,2,2,4, lwd=4, col='red')
dev.off()

