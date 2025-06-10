#Comparador ciclos y resultados de FF de Pablo Tancredi
# Medidas a 300 kHz y distintos campos
#%% Librerias
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import chardet
import re
import os
from uncertainties import ufloat
#%% Funciones
def plot_ciclos_promedio(directorio):
    # Buscar recursivamente todos los archivos que coincidan con el patrón
    archivos = glob(os.path.join(directorio, '**', '*ciclo_promedio*.txt'), recursive=True)

    if not archivos:
        print(f"No se encontraron archivos '*ciclo_promedio.txt' en {directorio} o sus subdirectorios")
        return
    fig,ax=plt.subplots(figsize=(8, 6),constrained_layout=True)
    for archivo in archivos:
        try:
            # Leer los metadatos (primeras líneas que comienzan con #)
            metadatos = {}
            with open(archivo, 'r') as f:
                for linea in f:
                    if not linea.startswith('#'):
                        break
                    if '=' in linea:
                        clave, valor = linea.split('=', 1)
                        clave = clave.replace('#', '').strip()
                        metadatos[clave] = valor.strip()

            # Leer los datos numéricos
            datos = np.loadtxt(archivo, skiprows=9)  # Saltar las 8 líneas de encabezado/metadatos

            tiempo = datos[:, 0]
            campo = datos[:, 3]  # Campo en kA/m
            magnetizacion = datos[:, 4]  # Magnetización en A/m

            # Crear etiqueta para la leyenda
            nombre_base = os.path.split(archivo)[-1].split('_')[1]
            #os.path.basename(os.path.dirname(archivo))  # Nombre del subdirectorio
            etiqueta = f"{nombre_base}"

            # Graficar

            ax.plot(campo, magnetizacion, label=etiqueta)

        except Exception as e:
            print(f"Error procesando archivo {archivo}: {str(e)}")
            continue

    plt.xlabel('Campo magnético (kA/m)')
    plt.ylabel('Magnetización (A/m)')
    plt.title(f'Comparación de ciclos de histéresis {os.path.split(directorio)[-1]}')
    plt.grid(True)
    plt.legend()  # Leyenda fuera del gráfico
    plt.savefig('comparativa_ciclos_'+os.path.split(directorio)[-1]+'.png',dpi=300)
    plt.show()

def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 6 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                match = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = match.group(1)[2:]
                    value = float(match.group(2))
                    meta[key] = value
                else:
                    # Capturar los casos con nombres de archivo en las últimas dos líneas
                    match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                    if match_files:
                        key = match_files.group(1)[2:]  # Obtener el nombre de la clave sin '# '
                        value = match_files.group(2)     # Obtener el nombre del archivo
                        meta[key] = value

    # Leer los datos del archivo
    data = pd.read_table(path, header=17,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N

#LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata

#%% Localizo ciclos y resultados
ciclos_3A=glob(('3A_300/**/*ciclo_promedio*'),recursive=True)
res_3A=glob('3A_300/**/*resultados*',recursive=True)
ciclos_3A.sort()
res_3A.sort()

ciclos_3Z=glob(('3Z_300/**/*ciclo_promedio*'),recursive=True)
res_3Z=glob('3Z_300/**/*resultados*',recursive=True)
ciclos_3Z.sort()
res_3Z.sort()

ciclos_4Z=glob(('4Z_300/**/*ciclo_promedio*'),recursive=True)
res_4Z=glob('4Z_300/**/*resultados*',recursive=True)
ciclos_4Z.sort()
res_4Z.sort()

ciclos_5A=glob(('5A_300/**/*ciclo_promedio*'),recursive=True)
res_5A=glob('5A_300/**/*resultados*',recursive=True)
ciclos_5A.sort()
res_5A.sort()

ciclos_7A=glob(('7A_300/**/*ciclo_promedio*'),recursive=True)
res_7A=glob('7A_300/**/*resultados*',recursive=True)
ciclos_7A.sort()
res_7A.sort()

ciclos_7ASiO2=glob(('7ASiO2_300/**/*ciclo_promedio*'),recursive=True)
res_7ASiO2=glob('7ASiO2_300/**/*resultados*',recursive=True)
ciclos_7ASiO2.sort()
res_7ASiO2.sort()

ciclos_8A=glob(('8A_300/**/*ciclo_promedio*'),recursive=True)
res_8A=glob('8A_300/**/*resultados*',recursive=True)
ciclos_8A.sort()
res_8A.sort()

#%% Ciclos
labels=['3A','3Z','4Z','5A','7A','7A@SiO$_2$','8A']
labels2=['57 kA/m','47 kA/m','38 kA/m','29 kA/m','20 kA/m']

#%%3A
fig, ax = plt.subplots(nrows=1,figsize=(6,4.5),constrained_layout=True)
for i,c in enumerate(ciclos_3A):
    _,_,_,H,M,_ = lector_ciclos(c)
    ax.plot(H/1000,M,label=labels2[i])
    # H_max = (idc/10*float(meta_citrato['pendiente_HvsI '])+float(meta_citrato['ordenada_HvsI ']))/1000
    # frec = meta_citrato['frecuencia']/1000
    # titulo=f'{H_max:.1f} kA/m - {frec:.1f} kHz'
ax.grid()
ax.set_xlabel('H (kA/m)')
ax.set_ylabel('M (A/m)')
ax.legend(ncol=1)
ax.set_title(f'{labels[0]} - 300 kHz')
    # ax.set_xlim(0,60e3)
    # ax.set_ylim(0,)
#plt.savefig('comparativa_HM_tancredi_png',dpi=400)

plt.savefig('3A_ciclo_HM.png',dpi=400)
plt.show()

#%%3Z 
fig, ax = plt.subplots(nrows=1,figsize=(6,4.5),constrained_layout=True)
for i,c in enumerate(ciclos_3Z[::-1]):
    _,_,_,H,M,_ = lector_ciclos(c)
    ax.plot(H/1000,M,label=labels2[i])
    # H_max = (idc/10*float(meta_citrato['pendiente_HvsI '])+float(meta_citrato['ordenada_HvsI ']))/1000
    # frec = meta_citrato['frecuencia']/1000
    # titulo=f'{H_max:.1f} kA/m - {frec:.1f} kHz'
ax.grid()
ax.set_xlabel('H (kA/m)')
ax.set_ylabel('M (A/m)')
ax.legend(ncol=1)
ax.set_title(f'{labels[1]} - 300 kHz')
    # ax.set_xlim(0,60e3)
    # ax.set_ylim(0,)
#plt.savefig('comparativa_HM_tancredi_png',dpi=400)
plt.savefig('3Z_ciclo_HM.png',dpi=400)
plt.show()

#%%4Z
fig, ax = plt.subplots(nrows=1,figsize=(6,4.5),constrained_layout=True)
for i,c in enumerate(ciclos_4Z):
    _,_,_,H,M,_ = lector_ciclos(c)
    ax.plot(H/1000,M,label=labels2[i])
    # H_max = (idc/10*float(meta_citrato['pendiente_HvsI '])+float(meta_citrato['ordenada_HvsI ']))/1000
    # frec = meta_citrato['frecuencia']/1000
    # titulo=f'{H_max:.1f} kA/m - {frec:.1f} kHz'
ax.grid()
ax.set_xlabel('H (kA/m)')
ax.set_ylabel('M (A/m)')
ax.legend(ncol=1)
ax.set_title(f'{labels[2]} - 300 kHz')
    # ax.set_xlim(0,60e3)
    # ax.set_ylim(0,)
#plt.savefig('comparativa_HM_tancredi_png',dpi=400)
plt.savefig('4Z_ciclo_HM.png',dpi=400)
plt.show()
#%%5A
fig, ax = plt.subplots(nrows=1,figsize=(6,4.5),constrained_layout=True)
for i,c in enumerate(ciclos_5A[::-1]):
    _,_,_,H,M,_ = lector_ciclos(c)
    ax.plot(H/1000,M,label=labels2[i])
    # H_max = (idc/10*float(meta_citrato['pendiente_HvsI '])+float(meta_citrato['ordenada_HvsI ']))/1000
    # frec = meta_citrato['frecuencia']/1000
    # titulo=f'{H_max:.1f} kA/m - {frec:.1f} kHz'
ax.grid()
ax.set_xlabel('H (kA/m)')
ax.set_ylabel('M (A/m)')
ax.legend(ncol=1)
ax.set_title(f'{labels[3]} - 300 kHz')
    # ax.set_xlim(0,60e3)
    # ax.set_ylim(0,)
#plt.savefig('comparativa_HM_tancredi_png',dpi=400)
plt.savefig('5A_ciclo_HM.png',dpi=400)
plt.show()

#%%7A
fig, ax = plt.subplots(nrows=1,figsize=(6,4.5),constrained_layout=True)
for i,c in enumerate(ciclos_7A):
    _,_,_,H,M,_ = lector_ciclos(c)
    ax.plot(H/1000,M,label=labels2[i])
    # H_max = (idc/10*float(meta_citrato['pendiente_HvsI '])+float(meta_citrato['ordenada_HvsI ']))/1000
    # frec = meta_citrato['frecuencia']/1000
    # titulo=f'{H_max:.1f} kA/m - {frec:.1f} kHz'
ax.grid()
ax.set_xlabel('H (kA/m)')
ax.set_ylabel('M (A/m)')
ax.legend(ncol=1)
ax.set_title(f'{labels[4]} - 300 kHz')
    # ax.set_xlim(0,60e3)
    # ax.set_ylim(0,)
#plt.savefig('comparativa_HM_tancredi_png',dpi=400)
plt.savefig('7A_ciclo_HM.png',dpi=400)
plt.show()


#%%7ASiO2
fig, ax = plt.subplots(nrows=1,figsize=(6,4.5),constrained_layout=True)
for i,c in enumerate(ciclos_7ASiO2[::-1]):
    _,_,_,H,M,_ = lector_ciclos(c)
    ax.plot(H/1000,M,label=labels2[i])
    # H_max = (idc/10*float(meta_citrato['pendiente_HvsI '])+float(meta_citrato['ordenada_HvsI ']))/1000
    # frec = meta_citrato['frecuencia']/1000
    # titulo=f'{H_max:.1f} kA/m - {frec:.1f} kHz'
ax.grid()
ax.set_xlabel('H (kA/m)')
ax.set_ylabel('M (A/m)')
ax.legend(ncol=1)
ax.set_title(f'{labels[5]} - 300 kHz')
    # ax.set_xlim(0,60e3)
    # ax.set_ylim(0,)
#plt.savefig('comparativa_HM_tancredi_png',dpi=400)
plt.savefig('7ASiO2_ciclo_HM.png',dpi=400)
plt.show()

#%% 8A
fig, ax = plt.subplots(nrows=1,figsize=(6,4.5),constrained_layout=True)
for i,c in enumerate(ciclos_8A):
    _,_,_,H,M,_ = lector_ciclos(c)
    ax.plot(H/1000,M,label=labels2[i])
    # H_max = (idc/10*float(meta_citrato['pendiente_HvsI '])+float(meta_citrato['ordenada_HvsI ']))/1000
    # frec = meta_citrato['frecuencia']/1000
    # titulo=f'{H_max:.1f} kA/m - {frec:.1f} kHz'
ax.grid()
ax.set_xlabel('H (kA/m)')
ax.set_ylabel('M (A/m)')
ax.legend(ncol=1)
ax.set_title(f'{labels[6]} - 300 kHz')
    # ax.set_xlim(0,60e3)
    # ax.set_ylim(0,)
#plt.savefig('comparativa_HM_tancredi_png',dpi=400)
plt.savefig('8A_ciclo_HM.png',dpi=400)
plt.show()
#%% SAR vs Hmax
SAR_3A,err_SAR_3A,Hmax_3A = [],[],[]
for r in res_3A:
    meta_3A, _,_,_,_,_,Hmax,_,_,_,_,_,SAR,_,_=lector_resultados(r)
    SAR_3A.append(np.mean(SAR))
    err_SAR_3A.append(np.std(SAR))
    Hmax_3A.append(np.mean(Hmax))
    
SAR_3Z,err_SAR_3Z,Hmax_3Z = [],[],[]
for r in res_3Z:
    meta_3Z, _,_,_,_,_,Hmax,_,_,_,_,_,SAR,_,_=lector_resultados(r)
    SAR_3Z.append(np.mean(SAR))
    err_SAR_3Z.append(np.std(SAR))
    Hmax_3Z.append(np.mean(Hmax))

SAR_4Z,err_SAR_4Z,Hmax_4Z = [],[],[]
for r in res_4Z:
    meta_4Z, _,_,_,_,_,Hmax,_,_,_,_,_,SAR,_,_=lector_resultados(r)
    SAR_4Z.append(np.mean(SAR))
    err_SAR_4Z.append(np.std(SAR))
    Hmax_4Z.append(np.mean(Hmax))

SAR_5A,err_SAR_5A,Hmax_5A = [],[],[]
for r in res_5A:
    meta_5A, _,_,_,_,_,Hmax,_,_,_,_,_,SAR,_,_=lector_resultados(r)
    SAR_5A.append(np.mean(SAR))
    err_SAR_5A.append(np.std(SAR))
    Hmax_5A.append(np.mean(Hmax))

SAR_7A,err_SAR_7A,Hmax_7A = [],[],[]
for r in res_7A:
    meta_7A, _,_,_,_,_,Hmax,_,_,_,_,_,SAR,_,_=lector_resultados(r)
    SAR_7A.append(np.mean(SAR))
    err_SAR_7A.append(np.std(SAR))
    Hmax_7A.append(np.mean(Hmax))

SAR_7ASiO2,err_SAR_7ASiO2,Hmax_7ASiO2 = [],[],[]
for r in res_7ASiO2:
    meta_7ASiO2, _,_,_,_,_,Hmax,_,_,_,_,_,SAR,_,_=lector_resultados(r)
    SAR_7ASiO2.append(np.mean(SAR))
    err_SAR_7ASiO2.append(np.std(SAR))
    Hmax_7ASiO2.append(np.mean(Hmax))

SAR_8A,err_SAR_8A,Hmax_8A = [],[],[]
for r in res_8A:
    meta_8A, _,_,_,_,_,Hmax,_,_,_,_,_,SAR,_,_=lector_resultados(r)
    SAR_8A.append(np.mean(SAR))
    err_SAR_8A.append(np.std(SAR))
    Hmax_8A.append(np.mean(Hmax))

#%%
xticks=np.array([19.9,29.2,38.5,47.8,57.1])*1e3
xlabels=['19.9','29.2','38.5','47.8','57.1']
fig2, ax = plt.subplots(nrows=1,figsize=(8,4),constrained_layout=True,sharex=True)

ax.errorbar(x=Hmax_3A,y=SAR_3A,yerr=err_SAR_3A, capsize=7,fmt='.-',label=labels[0])
ax.errorbar(x=Hmax_3Z,y=SAR_3Z,yerr=err_SAR_3Z, capsize=7,fmt='.-',label=labels[1])
ax.errorbar(x=Hmax_4Z,y=SAR_4Z,yerr=err_SAR_4Z, capsize=7,fmt='.-',label=labels[2])
ax.errorbar(x=Hmax_5A,y=SAR_5A,yerr=err_SAR_5A, capsize=7,fmt='.-',label=labels[3])
ax.errorbar(x=Hmax_7A,y=SAR_7A,yerr=err_SAR_7A, capsize=7,fmt='.-',label=labels[5])
ax.errorbar(x=Hmax_7ASiO2,y=SAR_7ASiO2,yerr=err_SAR_7ASiO2, capsize=7,fmt='.-',label=labels[4] )
ax.errorbar(x=Hmax_8A,y=SAR_8A,yerr=err_SAR_8A, capsize=7,fmt='.-',label=labels[6])

ax.set_ylim(0,)
ax.set_ylabel('SAR (W/g)')
ax.legend(ncol=2)
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)
ax.grid()
ax.set_xlabel('H$_{max}$ (kA/m)')
plt.savefig('comparativa_SAR_tancredi.png',dpi=400)

#%%
sar_3A=ufloat(np.mean(SAR_3A),np.std(SAR_3A))
sar_3Z=ufloat(np.mean(SAR_3Z),np.std(SAR_3Z))
sar_4Z=ufloat(np.mean(SAR_4Z),np.std(SAR_4Z))
sar_5A=ufloat(np.mean(SAR_5A),np.std(SAR_5A))
sar_7ASiO2=ufloat(np.mean(SAR_7ASiO2),np.std(SAR_7ASiO2))
sar_7A=ufloat(np.mean(SAR_7A),np.std(SAR_7A))
sar_8A=ufloat(np.mean(SAR_8A),np.std(SAR_8A))

# %%

# Configuración del gráfico
x = [1/7,2/7,3/7,4/7,5/7,6/7,1]  # Posiciones de las barras
width = 0.12  # Ancho de las barras
delta= 0.051

fig2, ax = plt.subplots(nrows=1,figsize=(6,4.5),constrained_layout=True,sharex=True)

bar11 = ax.bar(x[0],sar_3A.n, width, yerr=sar_3A.s, capsize=7,  label=labels[0])
bar13 = ax.bar(x[1],sar_3Z.n, width, yerr=sar_3Z.s, capsize=7,  label=labels[1])
bar13 = ax.bar(x[2],sar_4Z.n, width, yerr=sar_4Z.s, capsize=7,  label=labels[2])
bar21 = ax.bar(x[3],sar_5A.n, width, yerr=sar_5A.s, capsize=7,label=labels[3])
bar23 = ax.bar(x[4],sar_7ASiO2.n, width, yerr=sar_7ASiO2.s, capsize=7,label=labels[4] )
bar23 = ax.bar(x[5],sar_7A.n, width, yerr=sar_7A.s, capsize=7,label=labels[5])
bar31 = ax.bar(x[6],sar_8A.n, width, yerr=sar_8A.s, capsize=7,label=labels[6])

ax.set_xticks(x)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_xticklabels(labels)
ax.set_xlabel('Muestra')
#ax.legend(ncol=1,title='Ferrogel')
ax.set_ylabel('SAR (w/g)')


# ax1.set_ylabel('Hc (kA/m)', fontsize=12)
# ax1.set_title('Coercitivo vs Frecuencia',loc='left', fontsize=13)
# ax2.set_title('Remanencia vs Frecuencia',loc='left', fontsize=13)
# ax2.set_xlabel('f (kHz)', fontsize=12)
plt.suptitle('SAR - 300 kHz - 57 kA/m')
plt.savefig('comparativa_SAR_tancredi.png',dpi=400)
plt.show()
# %%
