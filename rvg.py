import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import sqrt
import time
import sys

def factorial(n):
		x = 1
		for i in range(1,n+1):
			x *= i
		return x
        
def randomSeed():
    t = time.perf_counter()
    return int(10**9*float(str(t-int(t))[0:]))
     
def LCG(n=1):
    seed = randomSeed()
    U = np.zeros(n)
    multiplier = 16807
    mod=(2**31)-1
    x = (seed*multiplier+1)%mod
    U[0] = x/mod
    for i in range(1,n):
        x = (x*multiplier+1)%mod
        U[i] = x/mod
    return U

def PRN(low=0,high=1,n=1):
	return low+(high-low)*LCG(n=n)

class bernoulli():
	def pmf(x,p):
		return p**x*(1-p)**(1-x)
	def mean(p):
		return p
	def var(p):
		return p*(1-p)
	def stdv(p):
		return bernoulli.var(p)**(0.5)
	def dist(p,n=1):
		prn = PRN(low=0,high=1,n=n)
		return (prn<=p).astype(int)
	def graph(p=0.8,n=1000):
		b=bernoulli.dist(p,n=n)
		plt.hist(b,bins=30,edgecolor='k')
		plt.xticks(fontsize=15)
		plt.xlabel("Bernoulli Distribution")
		plt.title("Bernoulli Distribution", fontsize="18")
		plt.yticks(fontsize=15)
		cmap = plt.get_cmap('jet')
		medium =cmap(0.25)
		handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [medium, medium]]
		labels = [f'p = {p}',f'n = {n}']
		plt.legend(handles,labels)
		plt.show()

class binomial():
	def first(n,x):
		return factorial(n)/(factorial(x)*factorial(n-x))
	def pmf(x,n,p):
		return binomial.first(n,x)*(p**x)*((1-p)**(n-x))
	def mean(n,p):
		return n*p
	def var(n,p):
		return n*p*(1 - p)
	def stdv(n,p):
		return n*p*(1-p)
	def graph(p=0.75, n=100):
		r = list(range(n + 1))
		r = PRN(1,100,100).astype(int)
		dist = [binomial.pmf(x, n, p) for x in r]
		plt.bar(r, dist)
		plt.title("Binomial distribution", fontsize="18")
		plt.xlabel("Probability of students pass the exam")
		plt.title("Binomail Distribution", fontsize="18")
		plt.show()


class poisson():
	def pmf(lmbda, r):
		return np.exp(-1*lmbda) * (lmbda**r) / (factorial(r))
	def mean(lmbda):
		return lmbda
	def stdv(lmbda):
		return sqrt(lmbda)
	def graph(p=0.75,n=100):
		r = list(range(n + 1))
		X = PRN(1,10,10).astype(int)
		lmbda = 4
		dist = [poisson.pmf(lmbda,i) for i in X]
		fig, ax = plt.subplots(1, 1, figsize=(6, 4))
		ax.plot(X, dist, 'bo', ms=8, label='poisson pmf')
		plt.ylabel("Probability", fontsize="18")
		plt.xlabel("Number of events", fontsize="18")
		plt.title("Poisson Distribution", fontsize="18")
		ax.vlines(X, 0, dist, colors='b', lw=5, alpha=0.5)
		plt.show()

class geometric():
	def mean(p):
		return 1/p
	def var(p):
		return (1-p)/p**2
	def pmf(k,p):
		return ((1 - p)**(k - 1))*p
	def graph(p=0.75,n=10):
		dist = [geometric.pmf(k,p) for k in range(1,n+1)]
		fig, ax = plt.subplots(1, 1, figsize=(6, 4))
		ax.plot(range(1,n+1), dist, 'bo', ms=8, label='Geometric pmf')
		plt.ylabel("Probability of error", fontsize="18")
		plt.xlabel("No. of times code compiled", fontsize="18")
		plt.title("Geometric Distribution", fontsize="18")
		ax.vlines(range(1,n+1), 0, dist, colors='b', lw=5, alpha=0.5)
		plt.show()

class normal():
	def cdf(u1,u2,sigma=1.0,mu=0.0):
		Z0 = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
		Z1 = np.sqrt(-2*np.log(u1))*np.sin(2*np.pi*u2)
		Z0 = Z0*sigma+mu
		return Z0
	def graph(low=0,high=1):
		u1 = PRN(low,high,10000)
		u2 = PRN(low,high,10000)
		dist = normal.cdf(u1,u2)
		plt.hist(dist,bins=25,edgecolor='k')
		plt.ylabel("density of probability", fontsize="18")
		plt.xlabel("points", fontsize="18")
		plt.title("Normal Distribution", fontsize="18")
		plt.show()

class exponential():
	def inverse(lmbda=1, U=[]):
		return -(1/lmbda)*(np.log(1-U))
	def graph(low=0,high=1):
		U = PRN(low,high,1000)
		dist = exponential.inverse(np.mean(U),U)
		plt.hist(dist,bins=25,edgecolor='k')
		plt.ylabel("", fontsize="18")
		plt.xlabel("", fontsize="18")
		plt.title("Exponential Distribution", fontsize="18")
		plt.show()

class weibull():
	def inverse(lmbda=1, k=2, U=[]):
		return lmbda*(-1*np.log(1-U))**(1/k)
	def graph(lmbda=0,k=1):
		U = PRN(0,1,1000)
		dist = weibull.inverse(lmbda=lmbda,k=k, U=U)
		plt.hist(dist,bins=25,edgecolor='k')
		plt.ylabel("", fontsize="18")
		plt.xlabel("", fontsize="18")
		plt.title("Weibull Distribution", fontsize="18")
		plt.show()	


if __name__ == "__main__":
	if sys.argv[1] == 'bern':
		bernoulli.graph(float(sys.argv[2]),int(sys.argv[3]))
	if sys.argv[1] == 'bin':
		binomial.graph(float(sys.argv[2]),int(sys.argv[3]))
	if sys.argv[1] == 'pois':
		poisson.graph(float(sys.argv[2]),int(sys.argv[3]))
	if sys.argv[1] == 'geom':
		geometric.graph(float(sys.argv[2]),int(sys.argv[3]))
	if sys.argv[1] == 'norm':
		normal.graph(int(sys.argv[2]),int(sys.argv[3]))
	if sys.argv[1] == 'exp':
		exponential.graph(int(sys.argv[2]),int(sys.argv[3]))
	if sys.argv[1] == 'weib':
		weibull.graph(int(sys.argv[2]),float(sys.argv[3]))