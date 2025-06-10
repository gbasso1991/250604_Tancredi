#%% VSM FF Pablo Tancredi 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
from sklearn.metrics import r2_score 
from mlognormfit import fit3
from mvshtools import mvshtools as mt
import re

def lineal(x,m,n):
    return m*x+n
#%% Levanto Archivos
data_3A = np.loadtxt(os.path.join('data_VSM','3A.txt'), skiprows=12)
data_3Z = np.loadtxt(os.path.join('data_VSM','3Zb.txt'), skiprows=12)
data_4Z = np.loadtxt(os.path.join('data_VSM','4Z.txt'), skiprows=12)
data_5A = np.loadtxt(os.path.join('data_VSM','5A1.txt'), skiprows=12)
data_7A = np.loadtxt(os.path.join('data_VSM','7A.txt'), skiprows=12)
data_7ASiO2 = np.loadtxt(os.path.join('data_VSM','7ASiO2a.txt'), skiprows=12)
data_8A = np.loadtxt(os.path.join('data_VSM','8A.txt'), skiprows=12)

#%% Armo vectores
H_3A = data_3A[:, 0]  # Gauss
m_3A = data_3A[:, 1]  # emu
H_3Z = data_3Z[:, 0]  # Gauss
m_3Z = data_3Z[:, 1]  # emu
H_4Z = data_4Z[:, 0]  # Gauss
m_4Z = data_4Z[:, 1]  # emu
H_5A = data_5A[:, 0]  # Gauss
m_5A = data_5A[:, 1]  # emu
H_7A = data_7A[:, 0]  # Gauss
m_7A = data_7A[:, 1]  # emu
H_7ASiO2 = data_7ASiO2[:, 0]  # Gauss
m_7ASiO2 = data_7ASiO2[:, 1]  # emu
H_8A = data_8A[:, 0]  # Gauss
m_8A = data_8A[:, 1]  # emu

#%% Masas 
masa_3A=0.0513 #g
masa_3Z=0.0502 #g
masa_4Z=0.0504 #g
masa_5A=0.0502 #g
masa_7A=0.0505 #g
masa_7ASiO2=0.0492 #g
masa_8A=0.0496 #g

C_all = 10  #concentracion estimada en g/L = kg/m³
C_m_en_m = C_all/1000 # uso densidad del H2O 1000 g/L

#%% Normalizo momentos por masa 
# emu -> emu/g (de NP)

m_3A /= masa_3A*C_m_en_m  # emu/g
m_3Z /= masa_3Z*C_m_en_m  # emu/g
m_4Z /= masa_4Z*C_m_en_m  # emu/g
m_5A /= masa_5A*C_m_en_m  # emu/g
m_7A /= masa_7A*C_m_en_m  # emu/g
m_7ASiO2 /= masa_7ASiO2*C_m_en_m  # emu/g
m_8A /= masa_8A*C_m_en_m  # emu/g

#%% Generar señales anhisteréticas
H_anhist_3A, m_anhist_3A = mt.anhysteretic(H_3A, m_3A)
H_anhist_3Z, m_anhist_3Z = mt.anhysteretic(H_3Z, m_3Z)
H_anhist_4Z, m_anhist_4Z = mt.anhysteretic(H_4Z, m_4Z)
H_anhist_5A, m_anhist_5A = mt.anhysteretic(H_5A, m_5A)
H_anhist_7A, m_anhist_7A = mt.anhysteretic(H_7A, m_7A)
H_anhist_7ASiO2, m_anhist_7ASiO2 = mt.anhysteretic(H_7ASiO2, m_7ASiO2)
H_anhist_8A, m_anhist_8A = mt.anhysteretic(H_8A, m_8A)

#%% Grafico ciclos 
labels=['3A','3Z','4Z','5A','7A','7ASiO2','8A']
H=[H_3A,H_3Z,H_4Z,H_5A,H_7A,H_7ASiO2,H_8A]
m=[m_3A,m_3Z,m_4Z,m_5A,m_7A,m_7ASiO2,m_8A]

H_ah=[H_anhist_3A,H_anhist_3Z,H_anhist_4Z,H_anhist_5A,H_anhist_7A,H_anhist_7ASiO2,H_anhist_8A]
m_ah=[m_anhist_3A,m_anhist_3Z,m_anhist_4Z,m_anhist_5A,m_anhist_7A,m_anhist_7ASiO2,m_anhist_8A]
# paso los campos de G a A/m
H_Am=[h/(4*np.pi) for h in H]
H_ah_Am=[h/(4*np.pi) for h in H_ah]


for i, e in enumerate(labels):
    fig, ax = plt.subplots(figsize=(6,4.5),constrained_layout=True)
    ax.plot(H_Am[i], m[i], '.-', label=e)
    ax.plot(H_ah_Am[i], m_ah[i], '-', label=e+' Anhist')
    
    ax.legend(ncol=1)
    ax.grid()
    ax.set_ylabel('m (emu/g)')
    ax.set_title(e)
    plt.xlabel('H (kA/m)')
    plt.savefig('VSM_'+e+'.png',dpi=300)
    plt.show()


#%%
fig1, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
ax.plot(H_3A, m_3A, '.-', label='3A')
ax.plot(H_anhist_3A, m_anhist_3A, '-', label='3A anhisteretica')
ax.plot(H_3Z, m_3Z, '.-', label='3Z')
ax.plot(H_anhist_3Z, m_anhist_3Z, '-', label='3Z anhisteretica')
ax.plot(H_4Z, m_4Z, '.-', label='4Z')
ax.plot(H_anhist_4Z, m_anhist_4Z, '-', label='4Z anhisteretica')
ax.plot(H_5A, m_5A, '.-', label='5A')
ax.plot(H_anhist_5A, m_anhist_5A, '-', label='5A anhisteretica')
for a in [ax]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu/g)')
plt.xlabel('H (G)')
plt.show()

fig2, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
ax.plot(H_7A, m_7A, '.-', label='7A')
ax.plot(H_anhist_7A, m_anhist_7A, '-', label='7A anhisteretica')
ax.plot(H_7ASiO2, m_7ASiO2, '.-', label='7ASiO2')
ax.plot(H_anhist_7ASiO2, m_anhist_7ASiO2, '-', label='7ASiO2 anhisteretica')
ax.plot(H_8A, m_8A, '.-', label='8A')
ax.plot(H_anhist_8A, m_anhist_8A, '-', label='8A anhisteretica')

for a in [ax]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu/g)')
plt.xlabel('H (G)')
plt.show()

#%% Realizo fits en ciclos (c/ contribiucion diamag)
#3A Fail
fit_3A = fit3.session(H_anhist_3A, m_anhist_3A, fname='3A', divbymass=False)
fit_3A.fix('sig0')
fit_3A.fix('mu0')
fit_3A.free('dc')
fit_3A.fit()
fit_3A.update()
fit_3A.free('sig0')
fit_3A.free('mu0')
fit_3A.set_yE_as('sep')
fit_3A.fit()
fit_3A.update()
fit_3A.save()
fit_3A.print_pars()
H_3A_fit = fit_3A.X
m_3A_fit = fit_3A.Y
m_3A_sin_diamag = m_anhist_3A - lineal(H_anhist_3A, fit_3A.params['C'].value, fit_3A.params['dc'].value)

#%% 3Z
fit_3Z = fit3.session(H_anhist_3Z, m_anhist_3Z, fname='3Z', divbymass=False)
fit_3Z.fix('sig0')
fit_3Z.fix('mu0')
fit_3Z.free('dc')
fit_3Z.fit()
fit_3Z.update()
fit_3Z.free('sig0')
fit_3Z.free('mu0')
fit_3Z.set_yE_as('sep')
fit_3Z.fit()
fit_3Z.update()
fit_3Z.save()
fit_3Z.print_pars()
H_3Z_fit = fit_3Z.X
m_3Z_fit = fit_3Z.Y
m_3Z_sin_diamag = m_anhist_3Z - lineal(H_anhist_3Z, fit_3Z.params['C'].value, fit_3Z.params['dc'].value)

#%%4Z Fail
fit_4Z = fit3.session(H_anhist_4Z, m_anhist_4Z, fname='4Z', divbymass=False)
fit_4Z.fix('sig0')
fit_4Z.fix('mu0')
fit_4Z.free('dc')
fit_4Z.fit()
fit_4Z.update()
fit_4Z.free('sig0')
fit_4Z.free('mu0')
fit_4Z.set_yE_as('sep')
fit_4Z.fit()
fit_4Z.update()
fit_4Z.save()
fit_4Z.print_pars()
H_4Z_fit = fit_4Z.X
m_4Z_fit = fit_4Z.Y
m_4Z_sin_diamag = m_anhist_4Z - lineal(H_anhist_4Z, fit_4Z.params['C'].value, fit_4Z.params['dc'].value)
#%%5A
fit_5A = fit3.session(H_anhist_5A, m_anhist_5A, fname='5A', divbymass=False)
fit_5A.fix('sig0')
fit_5A.fix('mu0')
fit_5A.free('dc')
fit_5A.fit()
fit_5A.update()
fit_5A.free('sig0')
fit_5A.free('mu0')
fit_5A.set_yE_as('sep')
fit_5A.fit()
fit_5A.update()
fit_5A.save()
fit_5A.print_pars()
H_5A_fit = fit_5A.X
m_5A_fit = fit_5A.Y
m_5A_sin_diamag = m_anhist_5A - lineal(H_anhist_5A, fit_5A.params['C'].value, fit_5A.params['dc'].value)
#%% 7A
fit_7A = fit3.session(H_anhist_7A, m_anhist_7A, fname='7A', divbymass=False)
fit_7A.fix('sig0')
fit_7A.fix('mu0')
fit_7A.free('dc')
fit_7A.fit()
fit_7A.update()
fit_7A.free('sig0')
fit_7A.free('mu0')
fit_7A.set_yE_as('sep')
fit_7A.fit()
fit_7A.update()
fit_7A.save()
fit_7A.print_pars()
H_7A_fit = fit_7A.X
m_7A_fit = fit_7A.Y
m_7A_sin_diamag = m_anhist_7A - lineal(H_anhist_7A, fit_7A.params['C'].value, fit_7A.params['dc'].value)

#%% 7ASiO2 Fail
fit_7ASiO2 = fit3.session(H_anhist_7ASiO2, m_anhist_7ASiO2, fname='7ASiO2', divbymass=False)
fit_7ASiO2.fix('sig0')
fit_7ASiO2.fix('mu0')
fit_7ASiO2.free('dc')
fit_7ASiO2.fit()
fit_7ASiO2.update()
fit_7ASiO2.free('sig0')
fit_7ASiO2.free('mu0')
fit_7ASiO2.set_yE_as('sep')
fit_7ASiO2.fit()
fit_7ASiO2.update()
fit_7ASiO2.save()
fit_7ASiO2.print_pars()
H_7ASiO2_fit = fit_7ASiO2.X
m_7ASiO2_fit = fit_7ASiO2.Y
m_7ASiO2_sin_diamag = m_anhist_7ASiO2 - lineal(H_anhist_7ASiO2, fit_7ASiO2.params['C'].value, fit_7ASiO2.params['dc'].value)

#%% 8A
fit_8A = fit3.session(H_anhist_8A, m_anhist_8A, fname='8A', divbymass=False)
fit_8A.fix('sig0')
fit_8A.fix('mu0')
fit_8A.free('dc')
fit_8A.fit()
fit_8A.update()
fit_8A.free('sig0')
fit_8A.free('mu0')
fit_8A.set_yE_as('sep')
fit_8A.fit()
fit_8A.update()
fit_8A.save()
fit_8A.print_pars()
H_8A_fit = fit_8A.X
m_8A_fit = fit_8A.Y
m_8A_sin_diamag = m_anhist_8A - lineal(H_anhist_8A, fit_8A.params['C'].value, fit_8A.params['dc'].value)


#%% Graficar ciclos y fits 'A'

label=['5A','7A','8A','3A']
m=[m_5A,m_7A,m_8A,m_3A]
H=[H_5A,H_7A,H_8A,H_3A]


label_fit=['5A fit','7A fit','8A fit']
m_fit=[m_5A_fit,m_7A_fit,m_8A_fit]
H_fit=[H_5A_fit,H_7A_fit,H_8A_fit]

fig, (ax1,ax2) = plt.subplots(nrows=2,figsize=(8,6),constrained_layout=True)

for i, e in enumerate(label):
    ax1.plot(H[i], m[i], '.-', label=e)


for i, e in enumerate(label_fit):
    ax2.plot(H[i], m[i], '-', label=e)
    
for a in [ax1,ax2]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu/g)')
plt.xlabel('H (G)')
ax1.set_title('ciclos 3A - 5A - 7A - 8A',loc='left')
ax2.set_title('fits 5A - 7A - 8A',loc='left')
plt.savefig('ciclos_y_fits_A.png',dpi=300)
plt.show()


