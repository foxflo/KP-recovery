# map from genus g (int) to tuple (B, [params]) where B is a gxg Riemann matrix, and params[i] = (U,V,W) are wave parameters needed to produce KP solutions of genus g via tau function
data = {}

params = []

B1=I*matrix([[1,.5,.5],[.5,1,.5],[.5,.5,1]])

u = matrix([1,.3183098861,-2.235157591]).T
v = matrix([3.141592654,-12.07995574,-4.073991721]).T
w = matrix([-211.2584193,-27.9766914,89.04583104]).T
params.append((u,v,w))

u = matrix([1,.3183098861,-2.235157591]).T
v = matrix([3.141592654,14.07995574,-9.969917617]).T
w = matrix([-211.2584193,95.2989871,61.26193481]).T
params.append((u,v,w))

u = matrix([1,.3183098861, 3.517756039]).T
v = matrix([3.141592654,24.01148816,-15.46901917]).T
w = matrix([-308.3442121,-427.694752,-336.6803733]).T
params.append((u,v,w))

u = matrix([1,.3183098861, 3.517756039]).T
v = matrix([3.141592654,-22.01148816,37.57173223]).T
w = matrix([-308.3442121,-644.572919,-86.7317208]).T
params.append((u,v,w))

data[3] = [(B1,params)]

params = []

B2=I*matrix([[1,.05,.5],[.05,1,.5],[.5,.5,1]])

u = matrix([1,.3183098861, .03689271122]).T
v = matrix([3.141592654,.5938403748,-2.069403203]).T
w = matrix([-1.131284045,-13.71207103,-14.67484527]).T
params.append((u,v,w))

u = matrix([1,.3183098861, .03689271122]).T
v = matrix([3.141592654,1.406159625,2.301206944]).T
w = matrix([-1.131284045,-9.884106742,5.921169830]).T
params.append((u,v,w))

u = matrix([1,.3183098861, .1171304410]).T
v = matrix([3.141592654,1.353216148,-1.457233431]).T
w = matrix([-1.204476091,-6.213955894,-13.32870220]).T
params.append((u,v,w))

u = matrix([1,.3183098861, .1171304410]).T
v = matrix([3.141592654,.6467838518,2.193185697]).T
w = matrix([-1.204476091,-9.542939663,3.873492679]).T
params.append((u,v,w))

data[3].append((B2,params))

params = []

B3=I*matrix([[1,.5,.05],[.5,1,.5],[.05,.5,1]])

u = matrix([1,.3183098861, -4.162799139]).T
v = matrix([3.141592654,39.52904010,95.08109124]).T
w = matrix([-670.4497552,-1298.095748,-1018.181873]).T
params.append((u,v,w))

u = matrix([1,.3183098861, -4.162799139]).T
v = matrix([3.141592654,-37.52904010,-121.2367296]).T
w = matrix([-670.4497552,-1661.22395,-2037.555586]).T
params.append((u,v,w))

u = matrix([1,.3183098861, 3.118580728]).T
v = matrix([3.14159654,-18.16973955,2.960265456]).T
w = matrix([-331.6876670,-191.3917194,-258.3211289]).T
params.append((u,v,w))

u = matrix([1,.3183098861, 3.118580728]).T
v = matrix([3.141592654,20.16973955,16.63435515]).T
w = matrix([-331.6876670,-10.7211805,-193.8834993]).T
params.append((u,v,w))

data[3].append((B3,params))

params = []

B4=I*matrix([[1,.5,.5],[.5,1,.05],[.5,.05,1]])

u = matrix([1,.3183098861, -2.493532548]).T
v = matrix([3.141592654,19.01847285,19.24498804]).T
w = matrix([-310.7480043,-349.8770430,23.7302685]).T
params.append((u,v,w))

u = matrix([1,.3183098861, -2.493532548]).T
v = matrix([3.141592654,-17.01847285,-34.91231511]).T
w = matrix([-310.7480043,-519.6971488,-231.4800100]).T
params.append((u,v,w))

u = matrix([1,.3183098861, 3.188305470]).T
v = matrix([3.141592654,10.32877599,8.130735889]).T
w = matrix([-202.6663867,19.8182739,-287.6826297]).T
params.append((u,v,w))

u = matrix([1,.3183098861, 3.188305470]).T
v = matrix([3.141592654,-8.328775990,11.90197820]).T
w = matrix([-202.6663867,-68.1033686,-269.9110690]).T
params.append((u,v,w))

#put all the genus 3 examples into data[3]
data[3].append((B4,params))

#Turku's genus 4 example
"""
from riemann_theta.siegel_reduction import siegel_reduction
from sage.schemes.riemann_surfaces.riemann_surface import numerical_inverse
CC=ComplexField(prec=60)
Mr=matrix(CC,[[0,0,0,0,0,-0.93374410984034652321*10^-3,0,0],[0,0,-0.50666245690409068051*10^-3,0,0,0,0,0],[0,0,0.48631314501195486807*10^-3,0,-0.95702553554556061172*10^-3,-0.47851276777278030596*10^-3,0,-0.1560075447834912432*10^-4],[0,0,.4692709939683523057*10^-4,0,0.25435625721525425896*10^-3,0.12717812860762712953*10^-3,0,-0.34821045600892472010*10^-3]])
Mi=matrix(CC,[[-0.37028416877234222286*I*10^-3,0.74056833754468444440*I*10^-3,-0.37028416877234222228*I*10^-3,0,0.37028416877234222148*I*10^-3,0.14811366750893688902*I*10^-2,0,-0.3702841687723422213*I*10^-3],[0.26176452554671226358*I*10^-3,0,0,0.26176452554671226360*I*10^-3,0,-0.26176452554671226368*I*10^-3,-0.52352905109342452727*I*10^-3,0],[0.31137937119672627930*I*10^-3,0,-0.5223307835103526567*I*10^-4,0.25914629284569101390*I*10^-3,0.67499182074448782431*I*10^-3,0.36361244954776154525*I*10^-3,0,-0.57052566404241729312*I*10^-3],[0.83078409529516327796*I*10^-5,0,0.18946089133738471936*I*10^-3,0.19776873229033635220*I*10^-3,-0.17284520943148145376*I*10^-3,-0.18115305038443308654*I*10^-3,0,-0.20607657324328798498*I*10^-3]])
Phat,_=siegel_reduction(Mr+Mi)
Omega1=Phat[:,:4]
Omega2=Phat[:,4:]
Omega1i=numerical_inverse(Omega1)
#Riemann matrix is Omega below
Omega=Omega1i*Omega2
"""
Omega = I*matrix([[2.6842059466633000,
  1.0567299367653518,
  1.8053701078679904,
  -1.7730558358752623],
 [1.0567299367653518,
  1.7636048773113787,
  0.88476903851606918,
  -0.87883583879530953],
 [1.8053701078679904,
  0.88476903851606917,
  2.6901391463840596,
  -0.71632589910991045],
 [-1.7730558358752623,
  -0.87883583879530953,
  -0.71632589910991046,
  2.6842059466633000]])

params = []

params.append([matrix(param).T for param in [(0.080794694780481593*I, 0.25800148438389056*I, 0.075381043437756121*I, -0.037369057686818575*I),(-0.12543771440396951*I, -0.63460404682997414*I, -0.10536509324787834*I, 0.060715790300484507*I),(0.18945659693662858*I, 1.3458522662381688*I, 0.13036260803952847*I, -0.11501171407189972*I)]])

params.append([matrix(param).T for param in [(-0.34583571431308104*I, -0.21536332223932170*I, -0.43825596733887908*I, 2.1372807093331194*I),(-7.0834313752859384*I, -3.6762440816477880*I, -10.980346975064135*I, 62.006863910892012*I),(-343.12191071243486*I, -182.96994317171887*I, -525.84570976373886*I, 2787.3316283509824*I)]])

params.append([matrix(param).T for param in [(0.000038717912143073322947693411349*I,-0.00029406826559529547247135153870*I,-0.23719514190917664876177958586*I,0.41022519791501254178985196786*I),
(-0.000084954130817458454490191212391*I,-0.0012727782971213746950307352350*I,-0.75614134400045927606701217414*I, -0.36119528481558329231218669104*I),
(0.00029693151396157704936717897157*I,-0.0043344045232136395294329906291*I,-1.9264564008607507354211952983*I,0.89726858512318630704149277418*I)]])

params.append([matrix(param).T for param in [(0.00014080043701105976395300580107*I, -0.046246196497554558838556872769*I,-0.23832894079628041085876061731*I, 0.0015370774382667751452548431519*I),
 (-0.00022633349924917570041477378020*I, 0.064727429312897722875839242838*I,0.76161339023564293963043754586*I, -0.0056916769264789487698246756088*I),
 (0.00027228495613958030951621617528*I,  -0.067068142746483241574310581524*I, -1.9494264161553445826353485226*I, 0.018054045053831583107078383930*I)]])

params.append([matrix(param).T for param in [(0.058594536817011329282223219089*I,  -0.047022601034476114726183399100*I, -0.00032135895535387217448927175789*I, 0.0025858389569682126428858240405*I),
 (0.12301529364074212488187230844*I, -0.066651901586870277013412957960*I, -0.00053317091132923801519659769633*I, 0.0055146875828670484708395638195*I),
 (0.19554964553040631094713101931*I, -0.071782239451140829126464718066*I, -0.00067142986835657082490384848735*I, 0.0090878549034938330477699408132*I)]])

params.append([matrix(param).T for param in [(0.058250372132081280179586554672*I, -0.00097481111001151129111971619886*I, -0.00017013495593936885932274745340*I, 0.41209184076076624616481496471*I),
 (- 0.12205010882170642274513241920*I, 0.0021857376793652733032066822385*I, -0.00015949388044292377283701239362*I, 0.35620771075420568416916292764*I),
 (0.19269994516157341509502322400*I, -0.0037442998560032234484155503814*I,  -0.00098897518171569409691698572455*I, 0.91431139565779748348330797283*I)]])

data[4] = [(Omega, params)]