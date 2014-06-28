'(c) Stephanie Schmitt-Grohe and Martin Uribe'
'Date July 17, 2001, revised 22-Oct-2004'
'Translated to Python by Sarunas Girdenas (sg325@exeter.ac.uk) and Hyun Changi Yi, 2014'
'This file replicates asset pricing model as in Schmitt-Grohe and Uribe (2004)'

import sympy as smp
import numpy as np
import math as mt
from scipy import linalg
from datetime import datetime # import this to calculate script execution time
startTime=datetime.now()


#Define Parameters

XBAR, RHO, THETA, BETTA = smp.symbols('XBAR, RHO, THETA, BETTA')

# Define Variables

p, pp, a, ap = smp.symbols('p, pp, a, ap')

# Write equations e1 and e2

e1 = -p + BETTA * (1 + pp) * smp.exp(THETA * ap);

e2 = -ap + (1-RHO) * XBAR + RHO * a;

# Create function f
F = smp.Matrix([e1,e2])

# Define the vector of controls, y, and states, x
x  = smp.Matrix([a])
y  = smp.Matrix([p]) 
xp = smp.Matrix([ap])
yp = smp.Matrix([pp])

# Make f a function of the logarithm of the state and control vector

#F = F.subs([(c,smp.exp(c)), (cp,smp.exp(cp)),(k,smp.exp(k)), (kp,smp.exp(kp)), (a,smp.exp(a)),(ap,smp.exp(ap))])

# Anal Deriv Function MATLAB


#def anal_deriv(f,x,y,xp,yp,approx):
#	if args == 5:
#		approx = 2

nx  = np.shape(x)[0]
ny  = np.shape(y)[0]
nxp = np.shape(xp)[0]
nyp = np.shape(yp)[0]

n   = np.shape(F)[0]

# Compute second order derivatives

#### STEADY STATE SOLUTION ###

XBAR  = 0.0179 # steady state of growth rate of dividend process
RHO   = -0.139 # persistence parameter of growth rate of dividend process
THETA = -1.5   # curvature of single-period utility function
BETTA = 0.95   # subjective discount factor
sig   = 1      # parameter scaling the vector of innovations
eta   = 0.0348 # matrix multiplying the vector of innovations
# Steady-state price-divident ratio, p, is the only endogenous variable
p     = BETTA * mt.exp(THETA*XBAR)/(1-BETTA*mt.exp(THETA*XBAR))
a     = XBAR #growth rate of dividend process (exogenous shock)

# Next period variables

pp = p
ap = a

# Parameters Dictionary

Parameters       = ['XBAR', 'RHO', 'THETA', 'BETTA', 'sig']

Param_Values     = [0.0179, -0.139, -1.5, 0.95, 1]

Parameters_dic   = {}
for i in range(len(Parameters)):
	Parameters_dic[Parameters[i]] = Param_Values[i]

# Variables Dictionary

Variables         = ['p', 'pp', 'a', 'ap']

SS_Variables      = [p, pp, a, ap] # steady state values of variables

SS_Variables_dic  = {}
for i in range(len(Variables)):
	SS_Variables_dic[Variables[i]] = SS_Variables[i]


### Compute Analytical first and second order derivatives ###

V   = [x,y,xp,yp] # variables

Fx   = [[] for f in F]              # matrix of 1st derivatives
nFx  = [[] for f in F]              # matrix of numerical evaluation 1st order derivatives
Fxx  = [[[[] for w in V] for v in V] for f in F] # matrix of 2nd derivatives
nFxx = [[[[] for w in V] for v in V] for f in F] # matrix numerical evaluation of 2nd order derivatives

fxx  = [] # intermediary matrix for calculation
nfxx = [] # intermediary matrix for calculation

# Compute Fx & Fxx and nFx & nFxx

for i,f in enumerate(F):
	f = smp.Matrix([f])
	for j,v in enumerate(V):
		v = smp.Matrix([v])
		fx = f.jacobian(v)
		nfx = (fx.subs(Parameters_dic)).subs(SS_Variables_dic)
		Fx[i].append(fx)
		nFx[i].append(nfx)
		for p,v in enumerate(V):
			v = smp.Matrix([v])
			fxx = fx.jacobian(v)
			Fxx[i][j][p].append(fxx)
			nfxx = (fxx.subs(Parameters_dic)).subs(SS_Variables_dic)
			nFxx[i][j][p].append(nfxx)

#### Compute Matrices gx and hx that define the 1st order approximation to the solution

First_Derivatives = []
for j in range(len(V)):
	First_Derivatives.append([nFx[i][j] for i in range(len(F))])

nfxp = First_Derivatives[2]
nfyp = First_Derivatives[3]
nfx  = First_Derivatives[0]
nfy  = First_Derivatives[1]

nfxp = np.asarray(nfxp)

A  = np.concatenate((nfxp,nfyp),axis=1)
A  = (-1)*A
B  = np.concatenate((nfx,nfy),axis=1)
NK = np.shape(nfx)[1]

s, t, q, z = linalg.qz(-A, B)

stake = 1

slt = abs(np.diag(t)) < stake*abs(np.diag(s))
nk = sum(slt)

def qzdiv(stake, A, B, Q, Z):
	# translation of qzdiv by Chris Sims
	# Takes U.T. matrices A, B, orthonormal matrices Q,Z, rearranges them
	# so that all cases of abs(B(i,i)/A(i,i))>stake are in lower right 
	# corner, while preserving U.T. and orthonormal properties and Q'AZ' and
	# Q'BZ'.
	[n, jnk] = np.shape(A)
	root = np.asarray([abs(np.diag(A)), abs(np.diag(B))])
	root[0] = root[0] - 1 * (root[0] < 1.e-13) * (root[0] + root[1])
	root[1] = [(root[1][i] / root[0][i]) for i in range(len(root[1]))]
	# root[1] = root[1]/root[0]
	for i in range(n, 0, -1):
		m = 0
		for j in range(i, 0, -1):
			if root[1][j-1] > stake or root[1][j-1] < -0.1:
				m = j
				break
		if m == 0:
			return A, B, Q, Z
		if m == i:
			print 'm is equal to i'
		for k in range(m, i):
			return qzswitch(k, A, B, Q, Z)
			root[1][k-1], root[1][k] = root[1][k], root[1][k-1]

def qzswitch(i, A, B, Q, Z):
	#  translation of qzswitch by Chris Sims
	# Takes U.T. matrices A, B, orthonormal matrices Q,Z, interchanges
	# diagonal elements i and i+1 of both A and B, while maintaining
	# Q'AZ' and Q'BZ' unchanged.  Does nothing if ratios of diagonal elements
	# in A and B at i and i+1 are the same.  Aborts if diagonal elements of
	# both A and B are zero at either position.
	a, d, b, e, c, f = A[i-1][i-1], B[i-1][i-1], A[i-1][i], B[i-1][i], A[i][i], B[i][i]
	wz = np.asarray([c*e - f*b, c*d - f*a])
	xy = np.asarray([b*d - e*a, c*d - f*a])
	n = mt.sqrt(np.dot(wz, wz))
	m = mt.sqrt(np.dot(xy, xy))
	if n == 0:
		return A, B, Q, Z
	else:
		wz = wz/n # the original code uses inverse matrix division '\' like n\wz
		xy = xy/m
		wz = [wz, [-wz[1], wz[0]]]
		xy = [xy, [-xy[1], xy[0]]]
		# replace (j+1)th and (j+2)th rows of A by matrix multiplication of xy and itself
		A[[i-1,i], :] = np.dot(xy, A[[i-1,i], :])
		B[[i-1,i], :] = np.dot(xy, B[[i-1,i], :])
		A[:, [i-1,i]] = np.dot(A[:, [i-1,i]], wz)
		B[:, [i-1,i]] = np.dot(B[:, [i-1,i]], wz)
		Z[:, [i-1,i]] = np.dot(Z[:, [i-1,i]], wz)
		Q[[i-1,i], :] = np.dot(xy, Q[[i-1,i], :])
	return A, B, Q, Z

s, t, q, z = qzdiv(stake, s, t, q, z)

z21 = z[nk:, 0:nk]
z11 = z[0:nk,0:nk]

s11 = -s[0:nk, 0:nk] # Multiply by (-1)
t11 = t[0:nk, 0:nk]

if nk > NK:
	print 'The Equilibrium is Locally Indeterminate'
elif nk < NK:
	print 'No Local Equilibrium Exists'

if np.linalg.matrix_rank(z11) < nk:
 	print 'Invertibility condition violated'

z11i  = np.dot(np.linalg.inv(z11), np.eye(nk)) # compute the solution

gx = np.real(np.dot(z21,z11i))
hx = np.real(np.dot(z11,np.dot(np.dot(np.linalg.inv(s11),t11),z11i)))


# ### SECOND ORDER APPROXIMATION ### 
# gxx_hxx file in MATLAB

m  = 0 
nx = np.shape(hx)[0] # rows of hx and hxx
ny = np.shape(gx)[0] # rows of gx and gxx
n  = nx + ny         # length of F
ngxx = nx**2*ny      # no of elements in gxx

Q   = [[] for i in range(n*nx*(nx+1)/2)] 
q_small = [[] for i in range(n*nx*(nx+1)/2)]



for i in range(n):
	for j in range(nx):
		for k in range(j+1):
				# Computing Q and using second, third, fifth and seventh term of the system (from paper)
				for yp_i in range(ny):
   					for x_i in range(nx):
   						for x_j in range(nx):
   							if x_i == k and x_j == j:
   								Ind = 1
   							else:
   								Ind = 0   							
   							Q[m].append(nfyp[i][yp_i] * hx[x_i, k] * hx[x_j, j] + Ind*nfy[i][yp_i])
				# Computing Q and using third and seventh term of the system for hxx (from paper)
				for x_i in range(nx):
   					for x_j in range(nx):
   						for beta_i in range(nx):
   							if x_i == k and x_j == j:
   								Ind = 1
   							else:
   								Ind = 0
   							Q[m].append(Ind * (sum(nfyp[i][yp_i] * gx[yp_i, beta_i] for yp_i in range(ny)) 
   								+ nfxp[i][beta_i]))
   				# Computing q and using the rest of terms
				#First term
 				q_small[m] = (np.asarray(nFxx[i][3][3]).dot(gx.dot(hx[:,k])) + np.asarray(nFxx[i][1][3]).dot(gx[:,k])+ np.asarray(nFxx[i][3][2]).dot(hx[:,k]) + np.asarray(nFxx[i][3][0])[0][k]).T.dot(gx.dot(hx[:,j]))
  				# Fourth Term 
   				q_small[m] += (np.asarray(nFxx[i][1][3]).dot(gx.dot(hx[:,k])) + np.asarray(nFxx[i][1][1]).dot(gx[:,k]) + np.asarray(nFxx[i][1][2]).dot(hx[:,k]) + np.asarray(nFxx[i][1][0])[0][k]).T.dot(gx[:,j])
	   			# Sixth term
				q_small[m] += (np.asarray(nFxx[i][2][3]).T.dot(gx).dot(hx[:,k]) + np.asarray(nFxx[i][2][1]).T.dot(gx[:,k]) + np.asarray(nFxx[i][2][2]).reshape(nx,nx).dot(hx[:,k]) + np.asarray(nFxx[i][2][0][0][k])).T.dot(hx[:,j])
	   			# Eight term
	   			q_small[m] += (np.asarray(nFxx[i][0][3]).reshape(nx,ny)[j].dot(gx).dot(hx[:,k]) + np.asarray(nFxx[i][0][1]).reshape(nx,ny)[j].dot(gx[:,k]) + np.asarray(nFxx[i][0][2]).reshape(nx,nx)[j].dot(hx[:,k]) + np.asarray(nFxx[i][0][0]).reshape(nx,nx)[j][k])
	   			m = m+1

# Computing hxx and gxx

# Use temp subfunction

def temp(nx,ny):
	Ahxx = np.zeros((nx**3,nx**2*(nx+1)/2))
	Agxx = np.zeros((ny*nx**2,ny*nx*(nx+1)/2))
	mx = 0
	my = 0
	for k in range(nx):
		for j in range(k,nx):
			for i in range(nx):
				Ahxx[(j)*nx+i+(k)*nx*nx,mx] = 1
				Ahxx[(k)*nx+i+(j)*nx*nx,mx] = 1
				mx = mx + 1
			for i in range(ny):
				Agxx[(j)*ny+i+(k)*ny*nx,my] = 1
				Agxx[(k)*ny+i+(j)*ny*nx,my] = 1
				my = my + 1

	A = np.zeros(((nx+ny)*nx**2,(nx+ny)*nx*(nx+1)/2))
	A[0:ny*nx**2, 0:ny*nx*(nx+1)/2] = Agxx
	A[ny*nx**2:, ny*nx*(nx+1)/2:]  = Ahxx
	return A

A = temp(nx, ny)

# Compute the matrices

Qt = np.dot(Q,A)
xt = linalg.lstsq(-Qt,np.asarray(q_small))[0]
x  = np.dot(A,xt)

sg = [ny, nx, nx] #size of gxx
sh = [nx, nx, nx] #size of hxx


gxx = x[0:ngxx].reshape(sg)
hxx = x[ngxx:].reshape(sh)


# Computing gss and hss matrices. 

#ne = eta.shape[1] #number of exogenous shocks (columns of eta)

qss = [[] for i in range(n)]
Qh  = [[] for i in range(n)]
Qg  = [[] for i in range(n)]

for i in range(n):
 	# First Term
 	Qh[i] = np.asarray(nfyp[i]).dot(gx)
 	# Second Term
 	qss[i] = np.sum(np.diag((np.asarray(nFxx[i][3][3]).dot(gx).dot(eta)).T.dot(gx).dot(eta)))
 	# Third Term
 	qss[i] += np.sum(np.diag((np.asarray(nFxx[i][3][2]).dot(eta)).T.dot(gx).dot(eta)))
 	# Fourth Term
 	ans = np.dot(np.asarray(nfyp[i]),gxx.reshape(ny,nx**2))
 	qss[i] += np.sum(np.diag((np.dot(ans.reshape(nx,nx),eta)).T.dot(eta)))
 	# Fifth Term
 	Qg[i] = np.asarray(nfyp[i])
 	# Sixth Term
 	Qg[i] += np.asarray(nfy[i])
 	# Seventh Term
 	Qh[i] += np.asarray(nfxp[i])
 	# Eight Term
 	qss[i] += np.sum(np.diag(np.asarray(nFxx[i][2][3]).T.dot(gx).dot(eta).T.dot(eta)))
 	# Ninth Term
 	qss[i] += np.sum(np.diag(np.asarray(nFxx[i][2][2]).reshape(nx,nx).dot(eta).T.dot(eta)))

# Compute gss and hss

Qhh = []
Qgg = []
for i in range(n):
	Qgg.append(Qg[i][0])
	Qhh.append(Qh[i][0])

x = np.concatenate((Qgg,Qhh),axis=1)
x1 = -linalg.lstsq(x,np.asarray(qss))[0]
gss = x1[0:ny]
hss = x1[ny:]

# Print all the output
print 'gx = ', gx
print '=========='
print 'hx =', hx
print 'gxx =', gxx
print '==========='
print 'hxx =', hxx # note that hxx values are transposed as compared to original matlab code
print 'gss = ', gss
print '============'
print 'hss = ', hss

# Impulse Response Functions

# #Shocks
# t = 100
# sig = 0.9
# e = 1
# Shock = []

# for i in range(t):
# 	e = sig*e
# 	Shock.append(e)

# # Number of periods
# T = np.shape(Shock)[0]+1
# # Number of shocks
# ne = np.shape(eta)[1]
# # Columns of X and Y
# nx = np.shape(hx)[0]
# ny = np.shape(gx)[0]

# # Specify initial conditions which are steady state values
# X = []
# x0 = x_s_state
# X[0:nx] = np.asarray(x0).T

# # Decompose x into first and second order components
# xf = x0
# xs = 0*x0

# y = [[] for i in range(ny)]
# Y = [[] for i in range(T)]

# for t in range(T):
# 	# Construct y(t)
# 	for i in range(ny):
# 	 	y[i] = np.dot(gx[i],np.asarray(xf+xs)) + np.dot((0.5*np.asarray(xf).T),gxx[i]).dot(np.asarray(xf)) + 0.5 * np.dot(gss[i],sig**2)

# 	Y[t][0:ny] = np.asarray(y).T

# # Construct x(t+1)
# xff = []
# xss = []
# for t in range(T):
# 	xoldf = xf 
# 	xolds = xs
# 	if t < T:
# 		for i in range(nx):
# 			xf = np.dot(hx[i],np.asarray(xoldf)) + sig * np.dot(np.asarray(eta[i]),np.asarray(Shock[t]).T)
# 			xff.append(xf)









print 'Computation time:', datetime.now()-startTime, 'seconds.'