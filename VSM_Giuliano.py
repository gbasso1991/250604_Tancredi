#%% VSM -  Ferrosolidos (ferrotec + laurico)
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
#%% Clase Muestra
class Muestra():
    'Archivo VSM'
    def __init__(self, nombre_archivo, masa_sachet, masa_sachet_NP, concentracion_MNP, err_concentracion_MNP):
        self.nombre_archivo = nombre_archivo
        self.masa_sachet = masa_sachet
        self.masa_sachet_NP = masa_sachet_NP
        self.concentracion_MNP = concentracion_MNP
        self.err_concentracion_MNP = err_concentracion_MNP
#%% Agregar muestras manualmente (sin vectores previos)
muestras = []
muestras.append(Muestra('data_VSM/3A.txt',0.0583,0.1096,10,0.01)) 
muestras.append(Muestra('data_VSM/3Zb.txt',0.0614, 0.1116,0.85,0.01))         
muestras.append(Muestra('data_VSM/4Z.txt',0.0634,0.1138,10,0.01)) 
muestras.append(Muestra('data_VSM/5A1.txt',0.058, 0.1082,0.79,0.01))           
muestras.append(Muestra('data_VSM/7A.txt',0.0534,0.1039,10,0.01)) 
muestras.append(Muestra('data_VSM/7ASiO2.txt',0.0582, 0.1074,0.17,0.01))        
muestras.append(Muestra('data_VSM/8A.txt',0.0642,0.1138,10,0.01)) 
#%%
nombre_archivo  = [muestra.nombre_archivo for muestra in muestras]
masa_sachet = np.array([muestra.masa_sachet for muestra in muestras])
masa_sachet_NP = np.array([muestra.masa_sachet_NP for muestra in muestras])
concentracion_MNP = np.array([muestra.concentracion_MNP for muestra in muestras])
errores_concentracion = np.array([muestra.err_concentracion_MNP for muestra in muestras])
masa_FF = masa_sachet_NP-masa_sachet

# Definir listas para almacenar los valores
sus_diamag=[]
offset=[]
mean_mu=[]
std_dev=[]
mean_mu_mu=[]
mag_sat=[]
fit_sessions=[]
m_s_diamag_norm_all=[]
H_all=[]
#Iniciamnos sesión de ajuste (aun no se ajusta) y graficamos curvas originales normalizadas
for k in range(len(nombre_archivo)):
    #Se lee el archivo
    print(k,'-'*50)
    archivo=nombre_archivo[k]
    data = np.loadtxt (archivo, skiprows=12)
    (campo,momento) = (data[:,0],data[:,1])
    
    #Normalizamos por masa de FF
    magnetizacion_FF=momento/masa_FF[k] 
           
    #Armamos la curva anhisteretica
    campo_anhist,magnetizacion_FF_anhist = mt.anhysteretic(campo,magnetizacion_FF)
    
    # #Se inicia la sesión de ajuste con la curva anhisterética
    fit_sessions.append(fit3.session (campo_anhist,magnetizacion_FF_anhist, fname='anhi'))
    fit=fit_sessions[k]
    fit.label=archivo[:-4]
    fit.fix('sig0')
    fit.fix('mu0')
    fit.free('dc')
    fit.fit()
    fit.update()
    fit.free('sig0')
    fit.free('mu0')
    fit.set_yE_as('sep')
    fit.fit()
    fit.update()
    fit.save()
    # fit.print_pars()
    H_fit = fit.X
    m_fit = fit.Y
    m_fit_sin_diamag = m_fit - lineal(H_fit, fit.params['C'].value, fit.params['dc'].value) 
    m_fit_sin_diamag_norm = m_fit_sin_diamag/concentracion_MNP[k]*1000 #emu/g == Am²/kg
    
    m_sin_diamag = magnetizacion_FF_anhist - lineal(campo_anhist, fit.params['C'].value, fit.params['dc'].value)
    m_sin_diamag_norm = m_sin_diamag/concentracion_MNP[k]*1000 #emu/g == Am²/kg

    #Guardo valores en las listas
    param_deriv=fit.print_pars(par=True) #ahora es un diccionario 
    sus_diamag.append(fit.params['C'])
    mean_mu.append(param_deriv['mean-mu']) #mb
    std_dev.append(param_deriv['stddev']) #mb
    mean_mu_mu.append(param_deriv['<mu>_mu']) #mb
    mag_sat.append(param_deriv['m_s']/concentracion_MNP[k]*1000) #normalizo por masa de NP
    m_s_diamag_norm_all.append(m_sin_diamag_norm)
    H_all.append(campo_anhist)
    
    text=fr'''$<\mu$> = {param_deriv["mean-mu"]} mb
    std dev = {param_deriv['stddev']} mb
    $<\mu_\mu$> = {param_deriv["<mu>_mu"]} mb'''    

    #ploteo ciclo individual
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    ax.plot(campo_anhist,m_sin_diamag_norm,'o-',label=fit.label)
    ax.plot(H_fit,m_fit_sin_diamag_norm,'-',label='fit')
    ax.text(0.75,1/2,text,bbox=dict(facecolor='tab:blue', alpha=0.5),transform=ax.transAxes,va='center',ha='center')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Campo (G)')
    ax.set_ylabel('Momento (emu/g)')
    ax.set_title('Magnetización s/ señal diamagnética normalizada por masa de MNP')
    plt.savefig('m_sin_diamag_anhist_norm_'+fit.label+'.png',dpi=300)
    #plt.show()
        
# %%
# Graficar resultados eliminando comportamiento diamagnético
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 10), sharex=True, constrained_layout=True)

ax1.set_title('mom magnetico por masa de NP - 100', loc='left')
for i in range(0,4):
    ax1.plot(H_all[i],m_s_diamag_norm_all[i], '.-', label=nombre_archivo[i])

# Eje inferior: Datos de "97"
ax2.set_title('mom magnetico por masa de NP - 97', loc='left')
for i in range(4,8):
    ax2.plot(H_all[i],m_s_diamag_norm_all[i], '.-', label=nombre_archivo[i])
ax2.set_xlabel('H (G)')

for a in [ax1,ax2]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (Am²/kg)')
plt.savefig('Mom_magnetico_por_masa_NPM.png',dpi=400)
plt.show()

# %%
