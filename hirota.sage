"""
This code doesn't quite seem to work. It's not clear to me whether this is due to a bug somewhere in the tedious algebra below or numerical issues.
"""

#list all derivatives of a given total order, e.g. d^2x, dxdy, dy^2 for order 2 in 2 variables, with repeats
derivs = lambda order:tuple([tuple(sorted(k)) for k in product([j for j in range(g)],repeat=order)])
#list derivatives of orders up to 7
indices = [derivs(order) for order in range(1,7)]

#count number of times a particular derivative appears in the total derivative of given order
coeffs = defaultdict(int)
for order in indices:
    for partial in order:
        coeffs[partial]=coeffs[partial]+1

#list of lists of derivatives of given order withotu repeats
pure_indices = [[deriv for deriv in coeffs.keys() if len(deriv)==order] for order in range(1,7)]
#convert index of derivative variable to vector for Nils Bruin's package
deriv_to_list = {deriv:[int(temp==deriv) for temp in range(g)] for deriv in range(g)}

#returns constant value of u function computed from KP equation via Hirota's bilinear form for a fixed x,y at t=0.
def checkHirota(x,y,t=0):
    t_ = theta(x,y,t)

    #compute evaluations of derivatives of Riemann theta, and then get actual derivatives by scaling by coefficients of x,y, or t
    temp = [RiemannTheta(z(x,y,t),B,derivs=[deriv_to_list[deriv]]) for deriv in range(g)]
    tau_x = lambda x,y,t: sum([u[i]*temp[i] for i in range(g)])
    t_x = tau_x(x,y,t)
    tau_t = lambda x,y,t: sum([w[i]*temp[i] for i in range(g)])
    t_t = tau_t(x,y,t)
    tau_y = lambda x,y,t: sum([v[i]*temp[i] for i in range(g)])
    t_y = tau_y(x,y,t)

    ders = derivs(2)
    temp = [RiemannTheta(z(x,y,t),B,derivs=[deriv_to_list[index] for index in partial]) for partial in ders]
    tau_xx = lambda x,y,t: sum([u[ders[i][0]]*u[ders[i][1]]*temp[i] for i in range(len(ders))])
    t_xx = tau_xx(x,y,t)
    tau_xt = lambda x,y,t: sum([u[ders[i][0]]*w[ders[i][1]]*temp[i] for i in range(len(ders))])
    t_xt = tau_xt(x,y,t)
    tau_xy = lambda x,y,t: sum([u[ders[i][0]]*v[ders[i][1]]*temp[i] for i in range(len(ders))])
    t_xy = tau_xy(x,y,t)
    tau_yy = lambda x,y,t: sum([v[ders[i][0]]*v[ders[i][1]]*temp[i] for i in range(len(ders))])
    t_yy = tau_yy(x,y,t)

    ders = derivs(3)
    temp = [RiemannTheta(z(x,y,t),B,derivs=[deriv_to_list[index] for index in partial]) for partial in ders]
    tau_xxx = lambda x,y,t: sum([u[ders[i][0]]*u[ders[i][1]]*u[ders[i][2]]*temp[i] for i in range(len(ders))])
    t_xxx = tau_xxx(x,y,t)
    tau_xxt = lambda x,y,t: sum([u[ders[i][0]]*u[ders[i][1]]*w[ders[i][2]]*temp[i] for i in range(len(ders))])
    t_xxt = tau_xxt(x,y,t)
    tau_xyy = lambda x,y,t: sum([u[ders[i][0]]*v[ders[i][1]]*v[ders[i][2]]*temp[i] for i in range(len(ders))])
    t_xyy = tau_xyy(x,y,t)
    tau_xxy = lambda x,y,t: sum([u[ders[i][0]]*u[ders[i][1]]*v[ders[i][2]]*temp[i] for i in range(len(ders))])
    t_xxy = tau_xxy(x,y,t)

    ders = derivs(4)
    temp = [RiemannTheta(z(x,y,t),B,derivs=[deriv_to_list[index] for index in partial]) for partial in ders]
    tau_xxxx = lambda x,y,t: sum([u[ders[i][0]]*u[ders[i][1]]*u[ders[i][2]]*u[ders[i][3]]*temp[i] for i in range(len(ders))])
    t_xxxx = tau_xxxx(x,y,t)
    tau_xxxt = lambda x,y,t: sum([u[ders[i][0]]*u[ders[i][1]]*u[ders[i][2]]*w[ders[i][3]]*temp[i] for i in range(len(ders))])
    t_xxxt = tau_xxxt(x,y,t)
    tau_xxyy = lambda x,y,t: sum([u[ders[i][0]]*u[ders[i][1]]*v[ders[i][2]]*v[ders[i][3]]*temp[i] for i in range(len(ders))])
    t_xxyy = tau_xxyy(x,y,t)

    #can reduce computations for next two
    tau_xxxxx = lambda x,y,t:sum([coeffs[(i,j,k,l,m)]*u[i]*u[j]*u[k]*u[l]*u[m]*RiemannTheta(z(x,y,t),B,derivs=[deriv_to_list[i],deriv_to_list[j],deriv_to_list[k],deriv_to_list[l],deriv_to_list[m]]) for (i,j,k,l,m) in pure_indices[4]])
    t_xxxxx = tau_xxxxx(x,y,t)

    tau_xxxxxx = lambda x,y,t:sum([coeffs[(i,j,k,l,m,n)]*u[i]*u[j]*u[k]*u[l]*u[m]*u[n]*RiemannTheta(z(x,y,t),B,derivs=[deriv_to_list[i],deriv_to_list[j],deriv_to_list[k],deriv_to_list[l],deriv_to_list[m],deriv_to_list[n]]) for (i,j,k,l,m,n) in pure_indices[5]])
    t_xxxxxx = tau_xxxxxx(x,y,t)

    #evaluate Hirota's bilinear form to figure out undetermined constant present in KP solution
    u_ = 2*(t_*t_xx-t_x*t_x)/(t_*t_)
    u_x = (4*t_x*t_x*t_x-6*t_*t_x*t_xx+2*t_*t_*t_xxx)/(t_*t_*t_)
    u_xx = (-12*t_x*t_x*t_x*t_x+24*t_*t_x*t_x*t_xx-6*t_*t_*t_xx*t_xx-8*t_*t_*t_x*t_xxx+2*t_*t_*t_*t_xxxx)/(t_*t_*t_*t_)
    u_tx = (2*t_*t_*t_*t_xxxt-6*t_*t_*t_x*t_xxt+12*t_*t_x*t_x*t_xt-6*t_*t_*t_xx*t_xt-12*t_x*t_x*t_x*t_t+12*t_*t_x*t_xx*t_t-2*t_*t_*t_xxx*t_t)/(t_*t_*t_*t_)
    u_yy = (2*t_*t_*t_*t_xxyy-4*t_*t_*t_xy*t_xy-4*t_*t_*t_x*t_xyy-4*t_*t_*t_xxy*t_y+16*t_*t_xy*t_x*t_y-12*t_x*t_x*t_y*t_y+4*t_*t_xx*t_y*t_y+4*t_*t_x*t_x*t_yy-2*t_*t_*t_xx*t_yy)/(t_*t_*t_*t_)
    u_xxxx = (-240*t_x*t_x*t_x*t_x*t_x*t_x+720*t_*t_x*t_x*t_x*t_x*t_xx-540*t_*t_*t_x*t_x*t_xx*t_xx+60*t_*t_*t_*t_xx*t_xx*t_xx-240*t_*t_*t_x*t_x*t_x*t_xxx+240*t_*t_*t_*t_x*t_xx*t_xxx-20*t_*t_*t_*t_*t_xxx*t_xxx+60*t_*t_*t_*t_x*t_x*t_xxxx-30*t_*t_*t_*t_*t_xx*t_xxxx-12*t_*t_*t_*t_*t_x*t_xxxxx+2*t_*t_*t_*t_*t_*t_xxxxxx)/(t_*t_*t_*t_*t_*t_)
    return ((4*u_tx-6*u_x*u_x-6*u_*u_xx-u_xxxx-3*u_yy)/(6*u_xx))[0].real

#takes quite a while
#could probability benefit from some kind of vectorization...
#plot3d(checkKP,(-.2,.2),(-.2,.2))
