import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

outdir = "/export/storage_covidvaccine/Result/Demand" #same as in demest_assm.py



#=================================================================
ylim = (-0.25, 0.1)
#=================================================================
# plot coefficients by capacity
max_rank = 10
nsplits = 3
coefs = []
capacities = [10000, 12500, 15000, 20000, 30000, 50000]
for capacity in capacities:
    setting_tag = f"{capacity}_{max_rank}_{nsplits}q"
    coefsavepath = f"{outdir}/coefs/{setting_tag}_coefs"
    coef_i = np.load(coefsavepath + ".npy")[0]
    coefs.append(coef_i)

transposed_coefs = list(zip(*coefs))

plt.figure(dpi=200)
for (ii, line) in enumerate(transposed_coefs[::-1]):
    plt.plot(capacities, line, label=f'HPI Quantile {nsplits-ii}')

plt.axis([max(capacities), min(capacities), ylim[0], ylim[1]])
plt.xlabel('Capacity')
plt.title('Figure 1: coefficients by capacity')
plt.legend(bbox_to_anchor=(1.35, 1))
figpath = f"{outdir}/coefplots/bycapacity_rank{max_rank}_nsplits{nsplits}.png"
plt.savefig(figpath, bbox_inches='tight')
print(figpath)

#=================================================================
# plot coefficients by max rank 
capacity = 10000
nsplits = 3
coefs = []
max_ranks = [1, 10, 30, 50, 100, 200]
for max_rank in max_ranks:
    setting_tag = f"{capacity}_{max_rank}_{nsplits}q"
    coefsavepath = f"{outdir}/coefs/{setting_tag}_coefs"
    coef_i = np.load(coefsavepath + ".npy")[0]
    coefs.append(coef_i)

transposed_coefs = list(zip(*coefs))

plt.figure(dpi=200)
for (ii, line) in enumerate(transposed_coefs[::-1]):
    print(ii, nsplits-ii)
    plt.plot(max_ranks, line, label=f'HPI Quantile {nsplits-ii}')

plt.axis([min(max_ranks), max(max_ranks), ylim[0], ylim[1]])
plt.xlabel('Max rank')
plt.title('Figure 2: coefficients by max rank')
plt.legend(bbox_to_anchor=(1.35, 1))
figpath = f"{outdir}/coefplots/bymaxrank_capacity{capacity}_nsplits{nsplits}.png"
plt.savefig(figpath, bbox_inches='tight')
print(figpath)



#=================================================================
# plot coefficients over iterations
# capacity, max_rank, nsplits = 10000, 10, 3
capacity, max_rank, nsplits = 10000, 200, 3
setting_tag = f"{capacity}_{max_rank}_{nsplits}q"
coefsavepath = f"{outdir}/coefs/{setting_tag}_coefs"
coefs = []
for ii in range(101):
    try:
        coefs.append(np.load(coefsavepath + f"{ii}.npy")[0])
    except:
        print("Iterations reached:", ii)
        break

transposed_coefs = list(zip(*coefs))
plt.figure(dpi=200)
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
for i, line in enumerate(transposed_coefs[::-1]):
    plt.plot(line, label=f'HPI Quantile {nsplits-i}')


plt.axis([0, len(coefs)-1, ylim[0], ylim[1]])
plt.xlabel('Iteration')
plt.title('Figure 3: coefficients by iteration')
plt.legend(bbox_to_anchor=(1.35, 1))
figpath = f"{outdir}/coefplots/byiter_capacity{capacity}_maxrank{max_rank}_nsplits{nsplits}.png"
plt.savefig(figpath, bbox_inches='tight')
print(figpath)


#=================================================================

r0offers_vec = []
r0assms_vec = []
for capacity in  [10000, 12500, 15000, 20000, 30000, 50000]:
    readpath = f"/mnt/staff/zhli/demest_assm_{capacity}_10_3.out"
    f =  open(readpath, 'r')
    lines = f.readlines()
    r0offers = [str(ll) for ll in lines if str(ll).startswith("% Rank 0 offers:")]
    r0offers = [float(ll.split(": ")[1].strip()) for ll in r0offers]
    print("Capacity:", capacity)
    print("1st choice offer:", round(r0offers[-1], 3))
    r0assms = [str(ll) for ll in lines if str(ll).startswith("% Rank 0 assignments:")]
    r0assms = [float(ll.split(": ")[1].strip()) for ll in r0assms]
    print("1st choice assignment:", round(r0assms[-1], 3))
    r0offers_vec.append(r0offers)
    r0assms_vec.append(r0assms)

plt.figure(dpi=200)
for (ii, line) in enumerate(r0offers_vec[::-1]):
    plt.plot(capacities, line, label=f'HPI Quantile {nsplits-ii}')
    
plt.axis([max(capacities), min(capacities), 0, 1])
plt.xlabel('Capacity')
plt.title('Figure 4: 1st choice offers by capacity')
plt.legend(bbox_to_anchor=(1.35, 1))
figpath = f"{outdir}/coefplots/r0offers{max_rank}_nsplits{nsplits}.png"
plt.savefig(figpath, bbox_inches='tight')
print(figpath)


#=================================================================
# testing to see how many iterations reached
capacity, max_rank, nsplits = 10000, 100, 3
setting_tag = f"{capacity}_{max_rank}_{nsplits}q"
coefsavepath = f"{outdir}/coefs/{setting_tag}_coefs"
for ii in range(101):
    try:
        coef = np.load(coefsavepath + f"{ii}.npy")
    except:
        print("Iterations reached:", ii)
        break



