R =  RealField(100)
RealNumber = R

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

#Nils Bruin's RiemannTheta package
from riemann_theta.riemann_theta import RiemannTheta
from riemann_theta.siegel_reduction import siegel_reduction
from sage.schemes.riemann_surfaces.riemann_surface import numerical_inverse

#fft and numerical integration libraries, uncomment commented lines for older versions of scipy
from scipy.fft import fft, fftfreq
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.linalg import lstsq
from scipy.optimize import nnls

#data file containing various Riemann matrices and parameters U,V,W for producing KP solutions
#data is a map from genus to a tuple (Riemann matrix, [params])
#where B is a Riemann matrix and params_i is a list of tuples (u,v,w) of parameters
from data import data

#return solution to kp equation generated from riemann matrix B and wave parameters (U,V,W) = params
def kp_func(B, params, g):
    RT = RiemannTheta(B)
    (u,v,w) = params
    d = matrix([0]*g).T
    def z(x,y,t):
        return (x*u+y*v+t*w+d).T[0]

    #hardcoded constants for second derivative (of tau function with respect to x)
    derivs2 = [a for b in [[(i,j)  for j in range(i,g)] for i in range(g)] for a in b]
    coeffs2 = [2-int(i==j) for (i,j) in derivs2]
    u_vals2=[u[i]*u[j] for (i,j) in derivs2]
    
    #tau functions
    tau = lambda x,y,t:RT(z=z(x,y,t))
    tau_x = lambda x,y,t: sum([a*b for (a,b) in zip(u.T[0],RT(z=z(x,y,t),derivs=[[i] for i in range(g)]))])
    tau_xx = lambda x,y,t:sum([a*b*c for (a,b,c) in zip(coeffs2,u_vals2,RT(z=z(x,y,t),derivs=derivs2))])
    #taus = np.array([tau(RR(i),0,0).real() for i in interval]) #save tau (or tau_x, tau_xx) if desired

    #actual kp solution, by default we'll use time t=0
    def f(x,y,t=0):
        t_ = tau(x,y,t)
        return (2*(t_*tau_xx(x,y,t)-tau_x(x,y,t)^2)/t_^2).real()

    return f

#anonymous function for computing fast Fourier transforms
fourier = lambda in_data: 2.0/samples*np.abs(fft(in_data)[:samples//2])

#find locations of local maxima (i.e. peaks) of fourier transform, sorted by amplitude
def find_peaks(fft_proc):
    temp = []
    for i in range(1,len(fft_proc)-1):
        if fft_proc[i]>=fft_proc[i+1] and fft_proc[i]>=fft_proc[i-1]:
            temp.append((freqs[i], fft_proc[i]))
    return sorted(temp,key=lambda x:x[1])

#save evals of kp solution on various slices of xy(t=0) plane, e.g. y=0, x=0, x=y, x=.1y, etc.
#kpsol is kp solution
#prefix is string filename to save results in as sobj = sage object
def save_evals(kpsol, prefix, step=.01, radius=40.96, samples=2**13+1):
    interval = np.arange(-radius,-radius+(samples)*step,step)
    freqs = fftfreq(int(samples),d=step)[:samples//2]
    u_evals = np.array([kpsol(RR(x),0) for x in interval])
    save(u_evals, prefix + "u_evals_x")
    save(find_peaks(fourier(u_evals)), prefix + "fft_x")
    
    u_evals = np.array([kpsol(RR(x), .1*RR(x)) for x in interval])
    save(u_evals, prefix + "u_evals_yeqp1x")
    save(find_peaks(fourier(u_evals)), prefix + "fft_yeqp1x")
    
    u_evals = np.array([kpsol(RR(x), .01*RR(x)) for x in interval])
    save(u_evals, prefix + "u_evals_yeqp01x")
    save(find_peaks(fourier(u_evals)), prefix + "fft_yeqp01x")
    
    u_evals = np.array([kpsol(0,RR(y)) for y in interval])
    save(u_evals, prefix + "u_evals_y")
    save(find_peaks(fourier(u_evals)), prefix + "fft_y")
    
    u_evals = np.array([kpsol(.1*RR(y), RR(y)) for y in interval])
    save(u_evals, prefix + "u_evals_xeqp1y")
    save(find_peaks(fourier(u_evals)), prefix + "fft_xeqp1y")
    
    u_evals = np.array([kpsol(.01*RR(y), RR(y)) for y in interval])
    save(u_evals, prefix + "u_evals_xeqp01y")
    save(find_peaks(fourier(u_evals)), prefix + "fft_xeqp01y")
    
    u_evals = np.array([kpsol(RR(x),RR(x)) for x in interval])
    save(u_evals, prefix + "u_evals_xeqy")
    save(find_peaks(fourier(u_evals)), prefix + "fft_xeqy")
    

#numerically integrate twice the input numerical kp solution evaluation data to obtain candidate tau function
def num_int(num_eval):
    u_int = cumulative_trapezoid(num_eval,interval,initial=0)
    avg_val = trapezoid(u_int,interval)/81.92
    temp = [a-avg_val for a in u_int]
    return [exp(RR(i)) for i in (cumulative_trapezoid(temp,interval,initial=0))/2]


#compute all integer vectors of length g whose 1-norm is at most norm
#in general, this choice may be very far from optimal 
def enum_vecs(norm,g=3):
    #list of all of them
    temp = [i for i in product(range(-l,l+1),repeat=g) if sum(map(abs,i))<=l]
    out = []
    #remove negatives, because the corresponding terms in make_sys_exp have identical real parts, so we can speed up the computation by a factor of 2
    for vec in temp:
        if not(vec in out or tuple([-i for i in vec]) in out):
            out.append(vec)
    return map(lambda a:matrix(a),out)
    
#approximate tau as a finite sum of exponentials (the number of terms depends on l), and view each exponential e^(...) as a variable, so that tau = \sum_i e^(...)
#Then sample m points uniformly at random in [-50, 50] and evaluate both sides of the equation above, to obtain m linear equations
#output the matrix A for coefficients of the sums \sum_i e^(...) and the vector of evaluations of tau, so that we have the system Ab = tau and we are interested in numerically solving for b
#tau is the candidate tau function, which we obtain by numerically integrating the kp solution twice in the x-direction, using num_int
##tau_guess = num_int(u_evals_x)
#fft_guess = fourier(tau_guess)
#fft_y = fourier(u_evals_y)
#fft_t = fourier(u_evals_t)
def make_sys_exp(tau, m, l,g=3):
    vecs = list(enum_vecs(l,g=g))
    A = np.zeros((m,len(vecs)))
    tau_mat = np.zeros(m)
    for k in range(m):
        sample_x = (np.random.rand()-.5)*10
        tau_mat[k] = tau(sample_x,0,0)
        for (i,vec) in enumerate(vecs):
            if (vec == np.matrix((0,)*g)).all():
                A[k,i] = (e^(2*pi*I*(vec*np_u)[0,0]*sample_x)).real()
            else:
                A[k,i] = 2*(e^(2*pi*I*(vec*np_u)[0,0]*sample_x)).real()
    return (A,tau_mat)

#generate the quadratic form nBn^T and turn it into a vector (we think of these quadratic forms as linear polynomials in the entries of B)
def quad_form(vec,g=3):
    out = [0]*((g^2+g)//2)
    counter = 0
    for i in range(g):
        for j in range(i,g):
            if i==j:
                out[counter] = -float(pi)*vec[0,i]*vec[0,j]
            else:
                out[counter] = -2*float(pi)*vec[0,i]*vec[0,j]
            counter = counter + 1
    return out

#sample m random points for x values using vectors of norm at most l
def recover_riemann(m,l,g=3):
    (A,tau_mat)= make_sys_exp(m,l,g=g)
    #set up the first system Ab=tau, thinking of the exponentials themselves as the variables for a linear system 
    B_data = lstsq(A,tau_mat)[0] #try nnls sometime
    #discard auxiliary data, which we may want in the future...
    vecs = list(enum_vecs(l,g=g))
    (B,log_B) = [],[]
    #set up the second system Bx = log_B by taking the logs of the exponentials found above, and thinking of the quadratic forms appearing in the exponents as linear polynomials in the entries of B 
    for (i,vec) in enumerate(vecs):
        #drop any negative numbers produced, since exp:R-> R_+. In practice they have all seemed to be very small, so would be sensitive to numerical issues.
        if B_data[i] > 0:
            B.append(quad_form(vec,g=g))
            log_B.append(log(B_data[i]))
    
    temp = list(zip(*sorted(zip(B,log_B), key=lambda x:x[1])))
    eqns, rhs = temp[0],temp[1]
    out = []
    #try least squares with all prefixes of the equations found
    for i in range(len(eqns)):
        out.append(lstsq(eqns[-i:],rhs[-i:])+(i,))
    return sorted(list(filter(lambda x:x[2]==g*(g+1)/2,out)),key=lambda x:x[1])

"""
g=3
sols = reciver_riemann(300,4)
temp = sols[0][0]

#reconstruct B matrix from data
B = [[0]*i+sols[0 if i==0 else i*g-i+1:(i+1)*g-i] for i in range(g)]
for i in range(g):
    for j in range(i):
        B[i][j] = B[j][i]
B = I*matrix(B)
"""

#turn the upper (g choose 2) entries of a symmetric matrix into matrix form
#e.g. symm_mat([[1,2],[1]],2) = [1,2]
#                               [2,1]
def symm_mat(mat, g):
    out_mat = [[0 for i in range(g)] for i in range(g)]
    counter = 0
    for i in range(g):
        for j in range(g):
            if i <= j:
                out_mat[i][j] = mat[counter]
                counter = counter + 1
            else:
                out_mat[i][j] = out_mat[j][i]
    return out_mat
