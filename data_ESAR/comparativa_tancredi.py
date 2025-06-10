#Comparador ciclos y resultados de NE@citrato (NE250331C)
#%% Librerias
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import chardet
import re
import os
from uncertainties import ufloat
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
#%% Ploteo todos los ciclos 
plot_ciclos_promedio('ex_citrato')
plot_ciclos_promedio('ex_hierro')
plot_ciclos_promedio('in_situ')
#%% Comparo a mismos idc/campo  
idcs = [150,135,120,105,90,75,60,45,30]
for idc in idcs:
    ciclos = glob(os.path.join('**/*2025*','*'+str(idc)+'dA'+'*ciclo_promedio*'),recursive=True)

    _,_,_,H_citrato,M_citrato,meta_citrato = lector_ciclos(ciclos[0])
    _,_,_,H_hierro,M_hierro,meta_hierro = lector_ciclos(ciclos[1])
    _,_,_,H_insitu,M_insitu,meta_insitu = lector_ciclos(ciclos[2])
    H_max = (idc/10*float(meta_citrato['pendiente_HvsI '])+float(meta_citrato['ordenada_HvsI ']))/1000
    frec = meta_citrato['frecuencia']/1000
    titulo=f'{H_max:.1f} kA/m - {frec:.1f} kHz'


    fig, ax = plt.subplots(nrows=1,figsize=(6,5),constrained_layout=True)
    ax.plot(H_insitu/1000,M_insitu,c='tab:red',label='in situ')
    ax.plot(H_citrato/1000,M_citrato,c='tab:green',label='ex citrato')
    ax.plot(H_hierro/1000,M_hierro,c='tab:blue',label='ex hierro')
    ax.grid()
    ax.set_xlabel('H (kA/m)')
    ax.set_ylabel('M (A/m)')
    ax.set_title(titulo,fontsize=12)
    ax.legend(title='Ferrogel',ncol=1)
    # ax.set_xlim(0,60e3)
    # ax.set_ylim(0,)
    plt.savefig('comparativa_HM_tancredi_'+str(idc)+'.png',dpi=400)
    plt.show()

#%% ploteo los mismos que el paper 
idcs_paper = [30,105,150]
campos = []
fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(12,4),constrained_layout=True,sharex=True,sharey=True)

for i,idc in enumerate(idcs_paper):
    ciclos_paper = glob(os.path.join('**/*2025*','*'+str(idc)+'dA'+'*ciclo_promedio*'),recursive=True)

    _,_,_,H_citrato,M_citrato,meta_citrato = lector_ciclos(ciclos_paper[0])
    _,_,_,H_hierro,M_hierro,meta_hierro = lector_ciclos(ciclos_paper[1])
    _,_,_,H_insitu,M_insitu,meta_insitu = lector_ciclos(ciclos_paper[2])
    H_max = (idc/10*float(meta_citrato['pendiente_HvsI '])+float(meta_citrato['ordenada_HvsI ']))/1000
    campos.append(round(H_max,0))
    frec = meta_citrato['frecuencia']/1000
    titulo=f'{H_max:.1f} kA/m - {frec:.1f} kHz'

    ax[i].plot(H_insitu/1000,M_insitu,c='tab:red',label='in situ')
    ax[i].plot(H_citrato/1000,M_citrato,c='tab:green',label='ex citrato')
    ax[i].plot(H_hierro/1000,M_hierro,c='tab:blue',label='ex hierro')

    
# for c in ciclos_PEM:
#     _,_,_,H_kAm,M_Am,_ = lector_ciclos(c)
#     ax.plot(H_kAm,M_Am,c='tab:red',label='PEM')
    ax[i].grid()
    ax[i].set_xlabel('H (kA/m)')
    ax[i].set_title(titulo,fontsize=12)
    ax[i].legend(title='Ferrogel',ncol=1,loc='upper left')
    # ax.set_xlim(0,60e3)
    # ax.set_ylim(0,)
ax[0].set_ylabel('M (A/m)')
plt.savefig('comparativa_HM_tancredi_030_105_150.png',dpi=400)
plt.show()

#%% SAR
res_150=glob(os.path.join('**/*2025*','*150*'+'*resultados*'),recursive=True)
res_105=glob(os.path.join('**/*2025*','*105*'+'*resultados*'),recursive=True)
res_030=glob(os.path.join('**/*2025*','*030*'+'*resultados*'),recursive=True)
res_150.sort()
res_105.sort()
res_030.sort()

meta_150_exc, _,_,_,_,_,_,_,_,_,_,_,SAR_150_exc,_,_=lector_resultados(res_150[0])
meta_150_exh, _,_,_,_,_,_,_,_,_,_,_,SAR_150_exh,_,_=lector_resultados(res_150[1])
meta_150_ins, _,_,_,_,_,_,_,_,_,_,_,SAR_150_ins,_,_=lector_resultados(res_150[2])

meta_105_exc, _,_,_,_,_,_,_,_,_,_,_,SAR_105_exc,_,_=lector_resultados(res_105[0])
meta_105_exh, _,_,_,_,_,_,_,_,_,_,_,SAR_105_exh,_,_=lector_resultados(res_105[1])
meta_105_ins, _,_,_,_,_,_,_,_,_,_,_,SAR_105_ins,_,_=lector_resultados(res_105[2])

meta_030_exc, _,_,_,_,_,_,_,_,_,_,_,SAR_030_exc,_,_=lector_resultados(res_030[0])
meta_030_exh, _,_,_,_,_,_,_,_,_,_,_,SAR_030_exh,_,_=lector_resultados(res_030[1])
meta_030_ins, _,_,_,_,_,_,_,_,_,_,_,SAR_030_ins,_,_=lector_resultados(res_030[2])

sar_150_exc=ufloat(np.mean(SAR_150_exc),np.std(SAR_150_exc))
sar_150_exh=ufloat(np.mean(SAR_150_exh),np.std(SAR_150_exh))
sar_150_ins=ufloat(np.mean(SAR_150_ins),np.std(SAR_150_ins))

sar_105_exc=ufloat(np.mean(SAR_105_exc),np.std(SAR_105_exc))
sar_105_exh=ufloat(np.mean(SAR_105_exh),np.std(SAR_105_exh))
sar_105_ins=ufloat(np.mean(SAR_105_ins),np.std(SAR_105_ins))

sar_030_exc=ufloat(np.mean(SAR_030_exc),np.std(SAR_030_exc))
sar_030_exh=ufloat(np.mean(SAR_030_exh),np.std(SAR_030_exh))
sar_030_ins=ufloat(np.mean(SAR_030_ins),np.std(SAR_030_ins))

# %%

# Configuración del gráfico
x = [1/3,2/3,1]  # Posiciones de las barras
width = 0.05  # Ancho de las barras
delta= 0.051
fig2, ax = plt.subplots(nrows=1,figsize=(6,5),constrained_layout=True,sharex=True)

bar11 = ax.bar(x[0]-delta,sar_030_ins.n, width, yerr=sar_030_ins.s, capsize=7, color='tab:red', label='In situ')
bar13 = ax.bar(x[0],sar_030_exc.n, width, yerr=sar_030_exc.s, capsize=7, color='tab:green', label='Ex citrato')
bar13 = ax.bar(x[0]+delta,sar_030_exh.n, width, yerr=sar_030_exh.s, capsize=7, color='tab:blue', label='Ex hierro')

bar21 = ax.bar(x[1]-delta,sar_105_ins.n, width, yerr=sar_105_ins.s, capsize=7, color='tab:red')
bar23 = ax.bar(x[1],sar_105_exc.n, width, yerr=sar_105_exc.s, capsize=7, color='tab:green')
bar23 = ax.bar(x[1]+delta,sar_105_exh.n, width, yerr=sar_105_exh.s, capsize=7, color='tab:blue')

bar31 = ax.bar(x[2]-delta,sar_150_ins.n, width, yerr=sar_150_ins.s, capsize=7, color='tab:red')
bar33 = ax.bar(x[2],sar_150_exc.n, width, yerr=sar_150_exc.s, capsize=7, color='tab:green')
bar33 = ax.bar(x[2]+delta,sar_150_exh.n, width, yerr=sar_150_exh.s, capsize=7, color='tab:blue')

ax.set_xticks(x)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_xticklabels(campos)
ax.set_xlabel('H (kA/m)')
ax.legend(ncol=1,title='Ferrogel')
ax.set_ylabel('SAR (W/g)')


# ax1.set_ylabel('Hc (kA/m)', fontsize=12)
# ax1.set_title('Coercitivo vs Frecuencia',loc='left', fontsize=13)
# ax2.set_title('Remanencia vs Frecuencia',loc='left', fontsize=13)
# ax2.set_xlabel('f (kHz)', fontsize=12)
plt.suptitle('SAR - 268 kHz')
plt.savefig('comparativa_HM_tancredi_030_105_150.png',dpi=400)
plt.show()
# %%
