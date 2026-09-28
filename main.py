from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import re
import io
import json
from typing import List, Dict, Any, Union
import scipy.stats as stats
from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, Column, String, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from fastapi import APIRouter, HTTPException
import uuid
from passlib.context import CryptContext

load_dotenv()

app = FastAPI(title="Validación de Métodos Analíticos")

# --- Conexión a base de datos ---
# En local, si no defines DATABASE_URL en tu .env, cae de vuelta a SQLite.
# En producción (Render/Railway/Fly.io), define DATABASE_URL con la cadena de PostgreSQL (Supabase/Neon/etc).
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./reportes.db")

connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Hashing de contraseñas ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Roles válidos para el registro de usuarios ---
ROLES_VALIDOS = {"fisicoquimico", "metales", "cromatografia"}

class ReporteDB(Base):
    __tablename__ = "reportes"
    id = Column(String, primary_key=True, index=True)
    codigo_informe = Column(String, unique=True, index=True)
    parametro = Column(String, index=True)
    matriz = Column(String)
    
    datos_crudos = Column(JSON)
    linealidad = Column(JSON)
    exactitud = Column(JSON)
    precision = Column(JSON)
    robustez = Column(JSON)
    muestras = Column(JSON)
    limites = Column(JSON)
    outliers = Column(JSON)
    
    datos_completos = Column(JSON)
    fecha_exportacion = Column(DateTime, default=datetime.utcnow)


class UsuarioDB(Base):
    __tablename__ = "usuarios"
    id = Column(String, primary_key=True, index=True)
    nombre = Column(String)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    rol = Column(String, nullable=False)  # "fisicoquimico" | "metales" | "cromatografia"
    fecha_registro = Column(DateTime, default=datetime.utcnow)


# Forzar la creación de tablas al iniciar FastAPI
@app.on_event("startup")
def startup_db_client():
    Base.metadata.create_all(bind=engine)

# Esquema Pydantic para recibir los datos del frontend
class ReporteCreate(BaseModel):
    codigo_informe: str
    parametro: str
    matriz: str
    datos_completos: dict


class UsuarioRegistro(BaseModel):
    nombre: str
    email: str
    password: str
    rol: str  # "fisicoquimico", "metales" o "cromatografia"


class UsuarioLogin(BaseModel):
    email: str
    password: str


@app.post("/api/registro")
def registrar_usuario(datos: UsuarioRegistro):
    rol_normalizado = datos.rol.strip().lower()
    if rol_normalizado not in ROLES_VALIDOS:
        raise HTTPException(
            status_code=400,
            detail=f"Rol inválido. Debe ser uno de: {', '.join(sorted(ROLES_VALIDOS))}"
        )
    if len(datos.password) < 6:
        raise HTTPException(status_code=400, detail="La contraseña debe tener al menos 6 caracteres.")

    email_normalizado = datos.email.strip().lower()
    db = SessionLocal()
    try:
        existente = db.query(UsuarioDB).filter(UsuarioDB.email == email_normalizado).first()
        if existente:
            raise HTTPException(status_code=400, detail="Ya existe una cuenta registrada con ese correo.")

        nuevo_usuario = UsuarioDB(
            id=str(uuid.uuid4()),
            nombre=datos.nombre.strip(),
            email=email_normalizado,
            password_hash=pwd_context.hash(datos.password),
            rol=rol_normalizado,
        )
        db.add(nuevo_usuario)
        db.commit()
        return {"mensaje": "Usuario registrado exitosamente", "nombre": nuevo_usuario.nombre, "email": nuevo_usuario.email, "rol": nuevo_usuario.rol}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error al registrar usuario: {str(e)}")
    finally:
        db.close()


@app.post("/api/login")
def iniciar_sesion(datos: UsuarioLogin):
    email_normalizado = datos.email.strip().lower()
    db = SessionLocal()
    try:
        usuario = db.query(UsuarioDB).filter(UsuarioDB.email == email_normalizado).first()
        if not usuario or not pwd_context.verify(datos.password, usuario.password_hash):
            raise HTTPException(status_code=401, detail="Correo o contraseña incorrectos.")
        return {"nombre": usuario.nombre, "email": usuario.email, "rol": usuario.rol}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al iniciar sesión: {str(e)}")
    finally:
        db.close()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CUOTAS = {"MB": 20, "LCM": 20, "CCV": 20, "EA": 20, "RT": 20}
MATRICES_MUESTRAS = ["agua superficial", "agua subterranea", "ar domestica", "ar no domestica", "arenoso", "arcilloso","limoso"]

def sanear_nan(obj: Any) -> Any:
    """Recorre recursivamente el resultado y reemplaza NaN/Inf (no serializables
    en JSON estandar) por None, para evitar 500 al responder."""
    if isinstance(obj, dict):
        return {k: sanear_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanear_nan(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    return obj

def clasificar_muestra(codigo_raw: Any):
    if not isinstance(codigo_raw, str):
        return None, None
    c = codigo_raw.strip().lower()
    c = " ".join(c.split())
    
    matriz = None
    
    if "arenoso" in c: matriz = "arenoso"
    elif "arcilloso" in c: matriz = "arcilloso"
    elif "limoso" in c: matriz = "limoso"
    # "agua residual no domestica" con la matriz doméstica.
    elif "no domestica" in c or "ar no domestica" in c:
        matriz = "ar no domestica"
    elif "domestica" in c or "ar domestica" in c:
        matriz = "ar domestica"
    elif "subterranea" in c:
        matriz = "agua subterranea"
    elif "superficial" in c:
        matriz = "agua superficial"
    else:
        return None, None

    if "adicionado" in c: return matriz, "adicionada"
    if "duplicado" in c: return matriz, "duplicada"
    return matriz, "normal"

def es_simbolo_quimico(valor: Any) -> bool:
    if not isinstance(valor, str):
        return False
    v= valor .strip()
    if v =="RAS": return True
    return bool(re.fullmatch(r"[A-Z][a-z]{0,2}", v))

def es_parametro_valido(valor: Any, area_analisis: str = "metales") -> bool:
    """Valida el nombre de un parámetro en el archivo de configuración.
    Para 'metales' exige símbolos químicos cortos (Fe, Pb, RAS...).
    Para 'fisicoquimico' acepta cualquier nombre no vacío (pH, Conductividad,
    Turbidez, Dureza Total, etc.), ya que estos parámetros no siguen la
    nomenclatura de símbolos químicos."""
    if not isinstance(valor, str):
        return False
    v = valor.strip()
    if v == "" or v.lower() == "nan":
        return False
    if area_analisis == "fisicoquimico":
        return True
    return es_simbolo_quimico(v)

def extraer_simbolo_factor(texto: str) -> str:
    """Extrae el símbolo del factor (A-G o a-g) conservando mayúsculas/minúsculas."""
    if not isinstance(texto, str):
        return None
    s = texto.strip()
    match = re.search(r'[Ff][Aa][Cc][Tt][Oo][Rr]\s*([A-Ga-g])', s)
    if match:
        return match.group(1)
    if len(s) == 1 and s in "AaBbCcDdEeFfGg":
        return s
    return None

def procesar_hoja_robustez(contenido_bytes: bytes, area_analisis: str = "metales") -> Dict[str, Any]:
    """Procesa la pestaña 'Robustez' y 'Hoja1' del archivo Formato.xlsx (estandares.xlsx)."""
    resultados = {}
    try:
        df = pd.read_excel(io.BytesIO(contenido_bytes), sheet_name="Robustez", header=None)
        
        # Extraer valores teóricos de la Hoja1
        df_hoja1 = pd.read_excel(io.BytesIO(contenido_bytes), sheet_name="Hoja1", header=None)
        teoricos_robustez = {}
        for r in range(len(df_hoja1)):
            elem = str(df_hoja1.iloc[r, 0]).strip()
            val_e = df_hoja1.iloc[r, 4]  # Columna E
            if elem and pd.notna(val_e):
                try:
                    teoricos_robustez[elem] = float(val_e)
                except ValueError:
                    pass
    except Exception:
        return {}

    # 1. Extracción de Metadatos de Robustez (Simbología, Factores y Escenarios Youden)
    for r in range(1, len(df)):
        val_0 = str(df.iloc[r, 0]).strip()
        if val_0.lower() == "simbologia":
            parametro = "Desconocido"
            for c in range(df.shape[1]):
                val_param = df.iloc[r-1, c]
                if pd.notna(val_param) and str(val_param).strip() != "":
                    parametro = str(val_param).strip()
                    break
                    
            factores = {}
            curr_r = r + 1
            while curr_r < len(df):
                simbolo = df.iloc[curr_r, 0]
                factor_val = df.iloc[curr_r, 1]
                
                if pd.isna(simbolo) or str(simbolo).strip() == "":
                    pass
                elif str(simbolo).strip().lower() == "resultado" or "simbologia" in str(simbolo).strip().lower():
                    break
                    
                if pd.notna(simbolo):
                    letra = str(simbolo).strip()
                    if len(letra) == 1: 
                        factores[letra] = str(factor_val).strip() if pd.notna(factor_val) else ""
                
                curr_r += 1
                if curr_r > r + 30: 
                    break
                    
            c_simb2 = -1
            for c in range(1, df.shape[1]):
                val_c = str(df.iloc[r, c]).strip()
                if val_c.lower() == "simbologia":
                    c_simb2 = c
                    break
                    
            escenarios = []
            if c_simb2 != -1:
                for c_esc in range(c_simb2 + 1, df.shape[1]):
                    esc_id = df.iloc[r, c_esc]
                    if pd.notna(esc_id):
                        eid_str = str(esc_id).strip()
                        if eid_str.endswith(".0"): eid_str = eid_str[:-2]
                        if not eid_str: continue
                        
                        combinacion = []
                        for row_letra in range(r + 1, r + 8):
                            if row_letra < len(df):
                                letra_esc = df.iloc[row_letra, c_esc]
                                if pd.notna(letra_esc):
                                    combinacion.append(str(letra_esc).strip())
                        
                        resultado = ""
                        if r + 8 < len(df):
                            res_val = df.iloc[r + 8, c_esc]
                            if pd.notna(res_val):
                                resultado = str(res_val).strip()
                                
                        escenarios.append({
                            "id": eid_str,
                            "combinacion": combinacion,
                            "resultado": resultado
                        })
            
            if parametro != "Desconocido":
                resultados[parametro] = {
                    "factores": factores,
                    "escenarios": escenarios,
                    "valor_teorico": teoricos_robustez.get(parametro, 0.0),
                    "datos_crudos": {sym: [] for sym in "AaBbCcDdEeFfGg"}
                }

    # 2. Si es FISICOQUÍMICO: Extraer los datos crudos directamente desde la hoja 'Robustez'
    if area_analisis == "fisicoquimico":
        filas, columnas = df.shape
        pool_fq_robustez = {}
        
        for r in range(filas):
            for c in range(columnas):
                val_celda = str(df.iloc[r, c]).strip().upper() if pd.notna(df.iloc[r, c]) else ""
                if val_celda == "PARAMETRO":
                    # El nombre del parámetro está 2 columnas a la derecha (columna c + 2)
                    if c + 2 < columnas:
                        val_p = df.iloc[r, c + 2]
                        if pd.notna(val_p) and str(val_p).strip() != "":
                            p_nombre = str(val_p).strip()
                            p_key = p_nombre.lower()
                            
                            if p_key not in pool_fq_robustez:
                                pool_fq_robustez[p_key] = {
                                    "nombre_real": p_nombre,
                                    "datos": {sym: [] for sym in "AaBbCcDdEeFfGg"}
                                }
                            
                            # Comenzar a leer factores bajando 2 filas desde "PARAMETRO"
                            curr_r = r + 2
                            while curr_r < filas:
                                val_f = df.iloc[curr_r, c]
                                if pd.isna(val_f) or str(val_f).strip() == "":
                                    break
                                if "PARAMETRO" in str(val_f).strip().upper():
                                    break
                                    
                                simbolo = extraer_simbolo_factor(str(val_f))
                                if simbolo and simbolo in "AaBbCcDdEeFfGg":
                                    analista_raw = str(df.iloc[curr_r, c + 1]).strip() if (c + 1 < columnas and pd.notna(df.iloc[curr_r, c + 1])) else "Analista 1"
                                    analista = "Analista 1" if "1" in analista_raw else ("Analista 2" if "2" in analista_raw else analista_raw)
                                    
                                    # Columna + 3: Concentración / Valor
                                    val_conc = df.iloc[curr_r, c + 3] if c + 3 < columnas else None
                                    if pd.notna(val_conc):
                                        try:
                                            conc = float(str(val_conc).replace(",", "."))
                                            pool_fq_robustez[p_key]["datos"][simbolo].append({
                                                "fecha": "Sin Fecha",
                                                "valor": conc,
                                                "analista": analista
                                            })
                                        except ValueError:
                                            pass
                                curr_r += 1

        # Mapear y combinar los datos crudos extraídos en el diccionario de resultados
        for p_key, obj_p in pool_fq_robustez.items():
            nombre_real = obj_p["nombre_real"]
            
            # Buscar si el parámetro ya existe (coincidencia de clave insensible a mayúsculas)
            target_param = None
            for res_p in resultados.keys():
                if res_p.lower() == p_key:
                    target_param = res_p
                    break
            
            if not target_param:
                factores_def = {s: f"Factor {s}" for s in "AaBbCcDdEeFfGg"}
                resultados[nombre_real] = {
                    "factores": factores_def,
                    "escenarios": [],
                    "valor_teorico": teoricos_robustez.get(nombre_real, 0.0),
                    "datos_crudos": {sym: [] for sym in "AaBbCcDdEeFfGg"}
                }
                target_param = nombre_real
            
            for sym, lecturas in obj_p["datos"].items():
                if lecturas:
                    resultados[target_param]["datos_crudos"][sym].extend(lecturas)

    return resultados

def clasificar_codigo(codigo_raw: Any) -> str:
    c = str(codigo_raw).strip().upper()
    if c in ["MB", "LCM", "CCV", "EA", "RT"]:
        return c
    if c.startswith("CCV-"):
        return "CCV"
    if c.startswith("EA-"):
        return "EA"
    if "RT" in c or ("ESTANDAR" in c and "RT" in c):
        return "RT"
    return None

def extraer_fecha(nombre_archivo: str) -> str:
    match = re.match(r"^(\d{4})(\d{2})(\d{2})", nombre_archivo)
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else "Sin Fecha"

def detectar_y_eliminar_outliers(datos, alpha=0.05):
    """Aplica la prueba de Grubbs iterativamente para detectar y remover datos atípicos."""
    if len(datos) < 3:
        return datos, []
    
    outliers_log = []
    datos_limpios = list(datos)
    
    while True:
        valores = np.array([d['valor'] for d in datos_limpios])
        n = len(valores)
        if n < 3: 
            break
            
        mean = np.mean(valores)
        std = np.std(valores, ddof=1)
        if std == 0: 
            break
        
        abs_dev = np.abs(valores - mean)
        max_idx = np.argmax(abs_dev)
        G_calc = abs_dev[max_idx] / std
        
        t_crit = stats.t.isf(alpha / (2 * n), n - 2)
        G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt((t_crit**2) / (n - 2 + t_crit**2))
        
        if G_calc > G_crit:
            outlier_info = datos_limpios.pop(max_idx)
            outlier_info['G_calc'] = round(G_calc, 4)
            outlier_info['G_crit'] = round(G_crit, 4)
            outliers_log.append(outlier_info)
        else:
            break
            
    return datos_limpios, outliers_log

def calcular_exactitud_y_grubbs(lista_valores, teorico):
    """Calcula las métricas de exactitud y gestiona los datos atípicos."""
    if not lista_valores:
        return [], [], None
        
    valores = np.array([d["valor"] for d in lista_valores])
    n = len(valores)
    
    if n < 3:
        prom = float(np.mean(valores)) if n > 0 else 0
        stats_basico = {
            "n_inicial": n, "promedio_inicial": prom, "desviacion_inicial": float(np.std(valores, ddof=1)) if n>1 else 0,
            "g_min": 0, "g_max": 0, "g_critico": 0, "n_final": n,
            "promedio_final": prom, "desviacion_final": float(np.std(valores, ddof=1)) if n>1 else 0,
            "error_pct": (abs(prom - teorico) / teorico * 100) if teorico else 0, 
            "error_promedio": prom - teorico, 
            "recuperacion": (prom / teorico * 100) if teorico else 0
        }
        return lista_valores, [], stats_basico

    prom = float(np.mean(valores))
    std = float(np.std(valores, ddof=1))
    
    G_min = (prom - np.min(valores)) / std if std > 0 else 0
    G_max = (np.max(valores) - prom) / std if std > 0 else 0
    
    alpha = 0.05
    t_crit = stats.t.isf(alpha / (2 * n), n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt((t_crit**2) / (n - 2 + t_crit**2))
    
    limpios, out_log = detectar_y_eliminar_outliers(lista_valores, alpha)
    
    val_limpios = np.array([d["valor"] for d in limpios])
    prom_limpio = float(np.mean(val_limpios)) if len(val_limpios) > 0 else 0
    std_limpio = float(np.std(val_limpios, ddof=1)) if len(val_limpios) > 1 else 0
    
    error_pct = (abs(prom_limpio - teorico) / teorico * 100) if teorico != 0 else 0.0
    error_promedio = (prom_limpio - teorico)
    recuperacion = (prom_limpio / teorico * 100) if teorico != 0 else 0.0
    
    stats_grubbs = {
        "n_inicial": n,
        "promedio_inicial": round(prom, 4),
        "desviacion_inicial": round(std, 4),
        "g_min": round(G_min, 4),
        "g_max": round(G_max, 4),
        "g_critico": round(G_crit, 4),
        "n_final": len(limpios),
        "promedio_final": round(prom_limpio, 4),
        "desviacion_final": round(std_limpio, 4),
        "error_pct": round(error_pct, 4),
        "error_promedio": round(error_promedio, 4),
        "recuperacion": round(recuperacion, 4)
    }
    
    return limpios, out_log, stats_grubbs

def calcular_estadistica_precision(datos_a1, datos_a2):
    """Ejecuta normalidad (por analista), ANOVA y Kruskal-Wallis (alternativa no paramétrica)."""
    if not datos_a1 or not datos_a2:
        return None
    
    y1 = np.array([d['valor'] for d in datos_a1])
    y2 = np.array([d['valor'] for d in datos_a2])
    
    if len(y1) == 0 or len(y2) == 0:
        return None
        
   
    sw_stat1, sw_p1 = stats.shapiro(y1) if len(y1) >= 3 else (0.0, 1.0)
    sw_stat2, sw_p2 = stats.shapiro(y2) if len(y2) >= 3 else (0.0, 1.0)
    

    dp_stat1, dp_p1 = stats.normaltest(y1) if len(y1) >= 8 else (0.0, 1.0)
    dp_stat2, dp_p2 = stats.normaltest(y2) if len(y2) >= 8 else (0.0, 1.0)
    if np.isnan(dp_stat1) or np.isnan(dp_p1):
        dp_stat1, dp_p1 = 0.0, 1.0
    if np.isnan(dp_stat2) or np.isnan(dp_p2):
        dp_stat2, dp_p2 = 0.0, 1.0


    datos_combinados = np.concatenate((y1, y2))
    todos_identicos = len(y1) > 0 and len(y2) > 0 and np.all(datos_combinados == datos_combinados[0])

    if len(y1) > 0 and len(y2) > 0 and not todos_identicos:
        try:
            kw_stat, kw_p = stats.kruskal(y1, y2)
        except ValueError:
            kw_stat, kw_p = 0.0, 1.0
    else:
        kw_stat, kw_p = 0.0, 1.0
    

    min_len = min(len(y1), len(y2))
    matriz = np.vstack((y1[:min_len], y2[:min_len])) if min_len > 1 else np.array([])
    
    anova_res = {}
    if matriz.size > 0:
        gran_promedio = np.mean(matriz)
        promedios_analista = np.mean(matriz, axis=1)
        promedios_replica = np.mean(matriz, axis=0)
        
        a, b = matriz.shape
        SST = np.sum((matriz - gran_promedio)**2)
        SSA = b * np.sum((promedios_analista - gran_promedio)**2)
        SSB = a * np.sum((promedios_replica - gran_promedio)**2)
        SSE = SST - SSA - SSB
        
        df_A, df_B = a - 1, b - 1
        df_E = df_A * df_B
        
        MSA = SSA / df_A if df_A > 0 else 0
        MSB = SSB / df_B if df_B > 0 else 0
        MSE = SSE / df_E if df_E > 0 else 0
        
        F_A = MSA / MSE if MSE > 0 else 0
        F_B = MSB / MSE if MSE > 0 else 0
        
        p_A = stats.f.sf(F_A, df_A, df_E) if MSE > 0 else 1
        p_B = stats.f.sf(F_B, df_B, df_E) if MSE > 0 else 1
        
        anova_res = {
            "analista": {"SS": round(SSA, 4), "df": df_A, "MS": round(MSA, 4), "F": round(F_A, 4), "p": round(p_A, 4)},
            "replica": {"SS": round(SSB, 4), "df": df_B, "MS": round(MSB, 4), "F": round(F_B, 4), "p": round(p_B, 4)},
            "error": {"SS": round(SSE, 4), "df": df_E, "MS": round(MSE, 4)},
            "total": {"SS": round(SST, 4), "df": (a*b - 1)}
        }

    return {
        "normalidad": {
            "shapiro": {
                "analista_1": {"stat": round(sw_stat1, 4), "p": round(sw_p1, 4), "normal": bool(sw_p1 > 0.05)},
                "analista_2": {"stat": round(sw_stat2, 4), "p": round(sw_p2, 4), "normal": bool(sw_p2 > 0.05)}
            },
            "dagostino": {
                "analista_1": {"stat": round(dp_stat1, 4), "p": round(dp_p1, 4), "normal": bool(dp_p1 > 0.05)},
                "analista_2": {"stat": round(dp_stat2, 4), "p": round(dp_p2, 4), "normal": bool(dp_p2 > 0.05)}
            }
        },
        "no_parametrica": {
            "kruskal_stat": round(kw_stat, 4),
            "kruskal_p": round(kw_p, 4),
            "significativa": bool(kw_p < 0.05)
        },
        "anova": anova_res
    }
    
def procesar_hoja_fisicoquimico(contenido_bytes, fecha):
    """Extrae datos experimentales basándose en la ubicación de la palabra PARAMETRO.
    Detecta TODAS las apariciones de "PARAMETRO" en cualquier fila/columna de la
    hoja (no solo la primera de cada fila), y delimita cada bloque de datos usando
    la siguiente aparición de "PARAMETRO" en esa MISMA columna (o el final de la
    hoja si no hay otra). Así soporta bloques apilados verticalmente en la misma
    columna y bloques ubicados en columnas distintas (side-by-side o en cualquier
    otra posición de la hoja)."""
    try:
        # Se lee sin encabezados para iterar por índices numéricos (fila, columna)
        df = pd.read_excel(io.BytesIO(contenido_bytes), sheet_name="Datos", header=None)
    except Exception as e:
        return {}
 
    pool_fq = {}
    filas, columnas = df.shape
 
    # 1. Encontrar TODAS las apariciones de "PARAMETRO" en toda la hoja
    ocurrencias = []
    for r in range(filas):
        for c in range(columnas):
            val = str(df.iloc[r, c]).strip().upper()
            if "PARAMETRO" in val:
                ocurrencias.append((r, c))
 
    # 2. Agrupar las filas donde aparece "PARAMETRO", por columna, para saber
    #    dónde termina cada bloque (la siguiente aparición en la misma columna)
    filas_por_columna = {}
    for (r, c) in ocurrencias:
        filas_por_columna.setdefault(c, []).append(r)
    for c in filas_por_columna:
        filas_por_columna[c].sort()
 
    # 3. Procesar cada bloque de forma independiente
    for (row_idx, col_param) in ocurrencias:
        if col_param + 2 >= columnas:
            continue
 
        nombre_parametro = str(df.iloc[row_idx, col_param + 2]).strip()
        if nombre_parametro not in pool_fq:
            pool_fq[nombre_parametro] = {"MB": [], "LCM": [], "CCV": [], "EA": []}
 
        # Límite del bloque: la siguiente aparición de PARAMETRO en la misma
        # columna, o el final de la hoja si no hay otra
        siguientes = [f for f in filas_por_columna[col_param] if f > row_idx]
        fin_bloque = siguientes[0] if siguientes else filas
 
        # Sumar 2 filas para empezar a leer los estándares
        curr_row = row_idx + 2
 
        while curr_row < fin_bloque:
            celda_actual = str(df.iloc[curr_row, col_param]).strip().upper()
 
            if celda_actual == "" or celda_actual == "NAN":
                curr_row += 1
                continue
 
            # 4. Clasificar el estándar
            std_code = None
            if "BLANCO" in celda_actual:
                std_code = "MB"
            elif "LIMITE" in celda_actual:
                std_code = "LCM"
            elif "INTERMEDIO" in celda_actual or "CCV" in celda_actual:
                std_code = "CCV"
            elif "ALTO" in celda_actual or "EA" in celda_actual:
                std_code = "EA"
 
            if std_code and col_param + 3 < columnas:
                # 5. Columna + 1: Analista
                analista_raw = str(df.iloc[curr_row, col_param + 1]).strip()
                analista = "Analista 1" if "1" in analista_raw else "Analista 2"
 
                # (La Columna + 2 son las unidades, se omite de la extracción de valores)
 
                # 6. Columna + 3: Valores experimentales
                valor_raw = df.iloc[curr_row, col_param + 3]
                try:
                    valor = float(str(valor_raw).replace(",", "."))
                    pool_fq[nombre_parametro][std_code].append({
                        "valor": valor,
                        "fecha": fecha,
                        "analista": analista
                    })
                except ValueError:
                    pass
 
            curr_row += 1
 
    return pool_fq
    
def procesar_hoja_linealidad(contenido_bytes: bytes) -> Dict[str, Any]:
    """Procesa la pestaña 'Curvas' presente dentro del archivo Formato.xlsx."""
    try:
        df = pd.read_excel(io.BytesIO(contenido_bytes), sheet_name="Curvas", header=None)
    except Exception:
        return {}

    parametros_brutos = []
    current_param = None
    current_curve = []
    skip_rows = 0

    for idx, row in df.iterrows():
        if skip_rows > 0:
            skip_rows -= 1
            continue

        val_a = row.iloc[0]
        val_b = row.iloc[1] if len(row) > 1 else np.nan

        if pd.notna(val_a):
            if isinstance(val_a, str) and not str(val_a).replace('.', '', 1).replace('-', '', 1).isdigit():
                if current_param and len(current_curve) > 0:
                    parametros_brutos.append({"param": current_param, "data": current_curve})
                current_param = val_a.strip()
                current_curve = []
                skip_rows = 1
            else:
                try:
                    conc = float(val_a)
                    signal = float(val_b) if pd.notna(val_b) else 0.0
                    if current_param is not None:
                        current_curve.append((conc, signal))
                except ValueError:
                    pass

    if current_param and len(current_curve) > 0:
        parametros_brutos.append({"param": current_param, "data": current_curve})

    agrupado = {}
    for item in parametros_brutos:
        p = item["param"]
        if p not in agrupado:
            agrupado[p] = []
        agrupado[p].append(item["data"])

    resultados = {}
    for p, curvas in agrupado.items():
        pendientes = []
        for curva in curvas:
            x = np.array([pt[0] for pt in curva])
            y = np.array([pt[1] for pt in curva])
            if len(x) > 1:
                slope, _, _, _, _ = stats.linregress(x, y)
                pendientes.append(slope)

        concentrations = {}
        for curva in curvas:
            for conc, sig in curva:
                if conc not in concentrations:
                    concentrations[conc] = []
                concentrations[conc].append(sig)

        sorted_concs = sorted(concentrations.keys())
        avg_x = []
        avg_y = []
        tabla_datos = []

        for c in sorted_concs:
            sigs = concentrations[c]
            avg_sig = float(np.mean(sigs))
            avg_x.append(c)
            avg_y.append(avg_sig)

            tabla_datos.append({
                "concentracion": c,
                "señales": sigs,
                "promedio": round(avg_sig, 4)
            })

        if len(avg_x) > 1:
            slope_avg, intercept_avg, r_val, _, _ = stats.linregress(avg_x, avg_y)
            r2_val = r_val ** 2
        else:
            slope_avg, intercept_avg, r_val, r2_val = 0.0, 0.0, 0.0, 0.0

        promedio_pendientes = float(np.mean(pendientes)) if pendientes else 0.0
        desviacion_pendientes = float(np.std(pendientes, ddof=1)) if len(pendientes) > 1 else 0.0

        for row in tabla_datos:
            if slope_avg != 0:
                conc_calc = (row["promedio"] - intercept_avg) / slope_avg
            else:
                conc_calc = 0.0

            row["conc_calculada"] = round(conc_calc, 4)
            if row["concentracion"] != 0:
                err = (abs(conc_calc - row["concentracion"]) / row["concentracion"]) * 100
            else:
                err = 0.0
            row["error_pct"] = round(err, 2)

        signo = "+" if intercept_avg >= 0 else "-"
        resultados[p] = {
            "curvas_raw": curvas,
            "tabla": tabla_datos,
            "stats": {
                "pendientes_individuales": [round(s, 4) for s in pendientes],
                "promedio_pendientes_raw": slope_avg,
                "intercepto_raw": intercept_avg,
                "promedio_pendientes": round(promedio_pendientes, 4),
                "desviacion_pendientes": round(desviacion_pendientes, 4),
                "sensibilidad": f"{promedio_pendientes:.4f} ± {desviacion_pendientes:.4f}",
                "intercepto": round(intercept_avg, 4),
                "r": round(r_val, 5),
                "r2": round(r2_val, 5),
                "ecuacion": f"y = {slope_avg:.4f}x {signo} {abs(intercept_avg):.4f}"
            }
        }

    return resultados

client_genai = genai.Client() if os.getenv("GEMINI_API_KEY") else None

class EvaluacionIARequest(BaseModel):
    elemento: str
    datos_resumen: Dict[str, Any]

@app.post("/api/procesar-datos")
async def procesar_datos(
    archivos_qc: List[UploadFile] = File(...),
    archivo_config: UploadFile = File(...),
    tipo_analisis: str = Form("estandar"),
    area_analisis: str = Form("metales"),
    humedades_suelos: str = Form("{}")
):
    # JSON esperado: {"arenoso": {"pw": 12.5, "humedad": 12.5}, "arcilloso": {...}, ...}
    try:
        dic_humedades = json.loads(humedades_suelos) or {}
    except Exception:
        dic_humedades = {}

    MATRICES_SUELO = ["arenoso", "arcilloso", "limoso"]

    def obtener_humedad(matriz):
        """Humedad (pW en metales / Humedad en fisicoquimico) de una submatriz de suelo."""
        h_obj = dic_humedades.get(matriz, {}) if matriz else {}
        clave = "humedad" if area_analisis == "fisicoquimico" else "pw"
        try:
            return float(h_obj.get(clave, h_obj.get("pw", h_obj.get("humedad", 0.0))) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    try:
        contenido_config = await archivo_config.read()
        df_config = pd.read_excel(io.BytesIO(contenido_config), skiprows=1, header=None)
        
        teoricos = {}
        for _, row in df_config.iterrows():
            simbolo = str(row[0]).strip()
            if es_parametro_valido(simbolo, area_analisis):
                teoricos[simbolo] = {
                    "LCM": float(row[1]) if pd.notna(row[1]) else 0.0,
                    "CCV": float(row[2]) if pd.notna(row[2]) else 0.0,
                    "EA": float(row[3]) if pd.notna(row[3]) else 0.0,
                    "RT": float(row[5]) if len(row) > 5 and pd.notna(row[5]) else 0.0,
                }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo archivo de configuración: {str(e)}")
    
    linealidad_global = procesar_hoja_linealidad(contenido_config)
    robustez_global = procesar_hoja_robustez(contenido_config, area_analisis=area_analisis)

    elementos_detectados = set(teoricos.keys())
    pool = {elem: {"MB": [], "LCM": [], "CCV": [], "EA": [], "RT": [], "muestras": {m: {"normal": [], "adicionada": [], "duplicada": []} for m in MATRICES_MUESTRAS}} for elem in elementos_detectados}

    for archivo in archivos_qc:
        if not archivo.filename.endswith(('.xls', '.xlsx')):
            continue
            
        fecha = extraer_fecha(archivo.filename)
        contenido = await archivo.read()
        
        if area_analisis == "fisicoquimico":
            datos_extraidos_fq = procesar_hoja_fisicoquimico(contenido, fecha)
            # Volcar datos al pool principal
            for parametro, controles_fq in datos_extraidos_fq.items():
                # Asegurar que el parámetro exista en el archivo de configuración teórico
                if parametro in pool:
                    for tipo_ctrl, data_list in controles_fq.items():
                        pool[parametro][tipo_ctrl].extend(data_list)
                        
        else:
            try:
                df = pd.read_excel(io.BytesIO(contenido), engine='xlrd', header=None)
                col_b = df.iloc[:, 1] if df.shape[1] > 1 else None
                col_d = df.iloc[:, 3] if df.shape[1] > 3 else None
                col_f = df.iloc[:, 5] if df.shape[1] > 5 else None
                col_n = df.iloc[:, 13] if df.shape[1] > 13 else None  # Columna N para pesos
                col_u = df.iloc[:, 20] if df.shape[1] > 20 else None
                
                if col_b is None or col_d is None or col_f is None:
                    continue
                
                codigo_actual = None
                muestra_actual= None
                tipo_muestra_actual = None
                peso_actual = 1.0
                factor_robustez_actual = None
                # Fecha vigente para el bloque de control/muestra y para el bloque
                # de robustez, capturada solo en la fila donde se detecta el rótulo
                # en columna D (o "FACTOR X"); se reutiliza para todas las lecturas
                # del bloque hasta que aparezca un nuevo rótulo.
                fecha_bloque_actual = fecha
                fecha_bloque_robustez = fecha

                for idx in range(len(df)):
                    val_d = col_d.iloc[idx]

                    # Fecha de esta fila: se toma de la columna U; si está vacía,
                    # se usa como respaldo la fecha extraída del nombre del archivo.
                    # Solo se usa para "anclar" fecha_bloque_actual/fecha_bloque_robustez
                    # en la fila donde se detecta el rótulo correspondiente.
                    fecha_u = str(col_u.iloc[idx]).split()[0] if col_u is not None and pd.notna(col_u.iloc[idx]) else fecha

                    # Gestión dinámica del código actual y captura de peso
                    if pd.notna(val_d) and str(val_d).strip() != "":
                        nuevo_codigo = clasificar_codigo(val_d)
                        matriz_m, tipo_m = clasificar_muestra(val_d)
                        # Si el usuario eligio suelos, solo cuentan las submatrices de suelo;
                        # en cualquier otro tipo de analisis, las de suelo se ignoran.
                        if matriz_m:
                            es_suelo_m = matriz_m in MATRICES_SUELO
                            if (tipo_analisis == "suelos") != es_suelo_m:
                                matriz_m, tipo_m = None, None
                        
                        if nuevo_codigo or matriz_m:
                            # Anclar la fecha del bloque a la fila donde se detectó
                            # el rótulo en columna D (control, estándar, matriz, etc.)
                            fecha_bloque_actual = fecha_u
                            if nuevo_codigo:
                                codigo_actual = nuevo_codigo
                                muestra_actual = None
                                tipo_muestra_actual = None
                            else:
                                muestra_actual = matriz_m
                                tipo_muestra_actual = tipo_m
                                codigo_actual = None
                                
                                if tipo_analisis == "suelos" and col_n is not None:
                                    try:
                                        if idx + 2 < len(df):
                                            val_peso = str(col_n.iloc[idx + 2]).replace(",",".")
                                            peso_actual = float(val_peso)
                                    except:
                                        peso_actual = 1.0
                            factor_robustez_actual = None

                        elif tipo_analisis != "suelos":
                            codigo_actual = None
                            muestra_actual = None
                            tipo_muestra_actual = None

                    val_b = str(col_b.iloc[idx]).strip() if pd.notna(col_b.iloc[idx]) else ""
                    
                    val_d_str = str(val_d).strip() if pd.notna(val_d) else ""

                    # Detecta el encabezado "FACTOR X" / "Factor X" (mayúsculas/minúsculas variables
                    # en la palabra "Factor", pero la letra conserva su mayúscula/minúscula porque
                    # distingue el factor "A" del factor "a"). Este encabezado suele estar varias
                    # filas ANTES de la tabla "Element_Symbol" con las lecturas, así que se guarda
                    # como contexto activo hasta el siguiente marcador.
                    if val_d_str.upper().startswith("FACTOR "):
                        factor_robustez_actual = val_d_str[len("FACTOR "):].strip()
                        fecha_bloque_robustez = fecha_u

                    if (factor_robustez_actual and val_b in robustez_global
                            and factor_robustez_actual in robustez_global[val_b]["datos_crudos"]):
                        simbolo = factor_robustez_actual
                        
                        if simbolo in robustez_global[val_b]["datos_crudos"]:
                            try:
                                conc_raw = float(str(col_f.iloc[idx]).replace(",", "."))
                                
                                # Aplicar cálculo de concentración según matriz
                                if tipo_analisis == "suelos":
                                    denominador = peso_actual * ((100 + obtener_humedad(muestra_actual)) / 100)
                                    if denominador == 0: denominador = 1
                                    conc = (conc_raw * 100) / denominador
                                elif tipo_analisis == "aire":
                                    conc = conc_raw * 0.05 * 9
                                else:
                                    conc = conc_raw

                                robustez_global[val_b]["datos_crudos"][simbolo].append({
                                    "fecha": fecha_bloque_robustez,
                                    "valor": conc
                                })
                            except (ValueError, TypeError):
                                pass
                    
                    if (codigo_actual or muestra_actual) and es_simbolo_quimico(val_b) and val_b in pool:
                        try:
                            conc_raw = float(str(col_f.iloc[idx]).replace(",", "."))
                            
                            # Cálculo Matemático según Matriz
                            if tipo_analisis == "suelos":
                                humedad_usar = obtener_humedad(muestra_actual)
                                denominador = peso_actual * ((100 + humedad_usar) / 100)
                                if denominador == 0: 
                                    denominador = 1  # Evita división por cero
                                conc = (conc_raw * 100) / denominador
                            elif tipo_analisis == "aire":
                                conc = conc_raw * 0.05 * 9
                            else:
                                conc = conc_raw

                            # 1. Ruta para Controles de Calidad (MB, LCM, CCV, EA)
                            if codigo_actual:
                                if len(pool[val_b][codigo_actual]) < CUOTAS[codigo_actual]:
                                    idx_actual = len(pool[val_b][codigo_actual])
                                    analista = "Analista 1" if idx_actual < 10 else "Analista 2"
                                    
                                    pool[val_b][codigo_actual].append({
                                        "valor": conc,
                                        "fecha": fecha_bloque_actual,
                                        "analista": analista
                                    })
                            
                            # 2. Ruta para Muestras (normal, adicionada, duplicada)
                            elif muestra_actual:
                                lista_destino = pool[val_b]["muestras"][muestra_actual][tipo_muestra_actual]
                                if len(lista_destino) < 20: # Límite de 20 réplicas (10 por analista)
                                    idx_actual = len(lista_destino)
                                    analista = "Analista 1" if idx_actual < 10 else "Analista 2"
                                    
                                    lista_destino.append({
                                        "valor": conc,
                                        "fecha": fecha_bloque_actual,
                                        "analista": analista
                                    })

                        except (ValueError, TypeError):
                            pass

            except Exception:
                # Ignorar archivos QC que no puedan leerse o procesarse.
                continue

    GRUPO_2 = ["Ca", "Mg", "Na", "K"]  # Modifica esta lista si tu grupo 2 abarca otros elementos

    if tipo_analisis == "ras":
        # Pesos moleculares y estados de oxidación (ox/PM)
        factores_ras = {
            "Ca": 2 / 40.078,
            "Mg": 2 / 24.305,
            "Na": 1 / 22.9897,
            "K":  1 / 39.0983
        }
        
        # 1. Convertir mg/L a mmol(+)/L para los 4 elementos
        for elem, factor in factores_ras.items():
            if elem in pool:
                for tipo_ctrl in ["MB", "LCM", "CCV", "EA"]:
                    for item in pool[elem][tipo_ctrl]:
                        item["valor"] = item["valor"] * factor
                        
        # 2. Calcular RAS dinámicamente por cada réplica
        if "RAS" not in pool:
            pool["RAS"] = {"MB": [], "LCM": [], "CCV": [], "EA": [], "RT":[]}
            
        for tipo_ctrl in ["MB", "LCM", "CCV", "EA", "RT"]:
            # Usar la cantidad mínima de réplicas encontradas para evitar errores de índice
            len_min = min(len(pool.get("Na", {}).get(tipo_ctrl, [])),
                          len(pool.get("Ca", {}).get(tipo_ctrl, [])),
                          len(pool.get("Mg", {}).get(tipo_ctrl, [])))
                          
            for i in range(len_min):
                na_val = pool["Na"][tipo_ctrl][i]["valor"]
                ca_val = pool["Ca"][tipo_ctrl][i]["valor"]
                mg_val = pool["Mg"][tipo_ctrl][i]["valor"]
                
                # Fórmula RAS = Na / sqrt((Ca+Mg)/2)
                denominador = np.sqrt((ca_val + mg_val) / 2)
                ras_val = (na_val / denominador) if denominador > 0 else 0
                
                pool["RAS"][tipo_ctrl].append({
                    "valor": ras_val,
                    "fecha": pool["Na"][tipo_ctrl][i]["fecha"],
                    "analista": pool["Na"][tipo_ctrl][i]["analista"] # Hereda analista del sodio
                })

        # 3. Agrupar la linealidad de Ca, Mg y Na dentro del objeto RAS
        if "RAS" not in linealidad_global:
            linealidad_global["RAS"] = {}
        linealidad_global["RAS"]["es_ras_combinado"] = True
        for elem in ["Ca", "Mg", "Na"]:
            if elem in linealidad_global:
                linealidad_global["RAS"][elem] = linealidad_global[elem]
                
    outliers_globales = {}
    exactitud_global = {}
    GRUPO_2 = ["Ca", "Mg", "Na", "K"]

    for elem, controles in pool.items():
        outliers_globales[elem] = {}
        exactitud_global[elem] = {}
        vol_antes = 49
        vol_muestra = 50
        vol_adicionado = 1
        conc_patron = 1000 if elem in GRUPO_2 else 10
        
        for tipo in ["MB", "LCM", "CCV", "EA", "RT"]:
            teorico_val = teoricos.get(elem, {}).get(tipo, 0.0) if tipo != "MB" else 0.0
            limpios, logs, stats_grubbs = calcular_exactitud_y_grubbs(controles[tipo], teorico_val)

            # No se sobreescribe el pool con la lista "limpia" de Grubbs: se
            # conservan las réplicas completas y cada registro se marca con
            # es_atipico para que el analista decida si las excluye o no.
            ids_atipicos = {id(o) for o in logs}
            for item in controles[tipo]:
                item["es_atipico"] = id(item) in ids_atipicos

            outliers_globales[elem][tipo] = logs
            if tipo != "MB":
                exactitud_global[elem][tipo] = stats_grubbs

    resultados = {}
    for elem, controles in pool.items():
        mb_datos = controles["MB"]
        lcm_datos = controles["LCM"]
        ccv_datos = controles["CCV"]
        ea_datos = controles["EA"]
        rt_datos = controles["RT"]
        
        if not mb_datos or not lcm_datos:
            continue

        val_teorico_lcm = teoricos.get(elem, {}).get("LCM", 0.0)
        val_teorico_ccv = teoricos.get(elem, {}).get("CCV", 0.0)
        val_teorico_ea = teoricos.get(elem, {}).get("EA", 0.0)
        val_teorico_rt = teoricos.get(elem, {}).get("RT", 0.0)
        
        def calcular_stats_grupo(lista_valores, teorico=0.0):
            if not lista_valores:
                return {"promedio": 0, "desviacion": 0, "cv": 0, "error_pct": 0, "valores": []}
            arr = np.array([d["valor"] for d in lista_valores])
            prom = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            cv = (std / prom * 100) if prom != 0 else 0.0
            err = (abs(prom - teorico) / teorico * 100) if teorico != 0 else 0.0
            
            return {
                "promedio": round(prom, 4),
                "desviacion": round(std, 4),
                "cv": round(cv, 2),
                "error_pct": round(err, 2),
                "valores": arr.tolist()
            }

        # Filtrado previo por analista para todos los tipos de control
        mb_a1 = [d for d in mb_datos if d["analista"] == "Analista 1"]
        mb_a2 = [d for d in mb_datos if d["analista"] == "Analista 2"]
        
        lcm_a1 = [d for d in lcm_datos if d["analista"] == "Analista 1"]
        lcm_a2 = [d for d in lcm_datos if d["analista"] == "Analista 2"]
        
        ccv_a1 = [d for d in ccv_datos if d["analista"] == "Analista 1"]
        ccv_a2 = [d for d in ccv_datos if d["analista"] == "Analista 2"]
        
        ea_a1 = [d for d in ea_datos if d["analista"] == "Analista 1"]
        ea_a2 = [d for d in ea_datos if d["analista"] == "Analista 2"]
        
        rt_a1 = [d for d in rt_datos if d["analista"] == "Analista 1"]
        rt_a2 = [d for d in rt_datos if d["analista"] == "Analista 2"]

        # Procesamiento estadístico agrupado
        stats_mb_global = calcular_stats_grupo(mb_datos)
        stats_mb_a1 = calcular_stats_grupo(mb_a1)
        stats_mb_a2 = calcular_stats_grupo(mb_a2)

        stats_lcm_global = calcular_stats_grupo(lcm_datos, val_teorico_lcm)
        stats_lcm_a1 = calcular_stats_grupo(lcm_a1, val_teorico_lcm)
        stats_lcm_a2 = calcular_stats_grupo(lcm_a2, val_teorico_lcm)
        
        stats_ccv_global = calcular_stats_grupo(ccv_datos, val_teorico_ccv)
        stats_ccv_a1 = calcular_stats_grupo(ccv_a1, val_teorico_ccv)
        stats_ccv_a2 = calcular_stats_grupo(ccv_a2, val_teorico_ccv)

        stats_ea_global = calcular_stats_grupo(ea_datos, val_teorico_ea)
        stats_ea_a1 = calcular_stats_grupo(ea_a1, val_teorico_ea)
        stats_ea_a2 = calcular_stats_grupo(ea_a2, val_teorico_ea)
        
        stats_rt_global = calcular_stats_grupo(rt_datos, val_teorico_rt)
        stats_rt_a1 = calcular_stats_grupo(rt_a1, val_teorico_rt)
        stats_rt_a2 = calcular_stats_grupo(rt_a2, val_teorico_rt)

        lod_posible = round(3 * stats_mb_global["desviacion"], 4)
        loq_posible = round(10 * stats_mb_global["desviacion"], 4)

        resultados[elem] = {
            "teorico_lcm": val_teorico_lcm,
            "teorico_ccv": val_teorico_ccv,
            "teorico_ea": val_teorico_ea,
            "lod_posible": lod_posible,
            "loq_posible": loq_posible,
            "outliers": outliers_globales[elem],
            "exactitud": exactitud_global[elem],
            "mb": {
                "global": stats_mb_global,
                "analista_1": stats_mb_a1,
                "analista_2": stats_mb_a2,
                "raw": mb_datos
            },
            "lcm": {
                "teorico_lcm": val_teorico_lcm,
                "global": stats_lcm_global,
                "analista_1": stats_lcm_a1,
                "analista_2": stats_lcm_a2,
                "raw": lcm_datos
            },
            "ccv": {
                "teorico_CCV": val_teorico_ccv,
                "global": stats_ccv_global,
                "analista_1": stats_ccv_a1,
                "analista_2": stats_ccv_a2,
                "raw": ccv_datos 
            },
            "ea": {
                "teorico_EA": val_teorico_ea,
                "global": stats_ea_global,
                "analista_1": stats_ea_a1,
                "analista_2": stats_ea_a2,
                "raw": ea_datos
            },
            "rt": {
                "teorico_RT": val_teorico_rt,
                "global": stats_rt_global,
                "analista_1": stats_rt_a1,
                "analista_2": stats_rt_a2,
                "raw": rt_datos
            },
            "precision": {
                "lcm": calcular_estadistica_precision(lcm_a1, lcm_a2),
                "ccv": calcular_estadistica_precision(ccv_a1, ccv_a2),
                "ea": calcular_estadistica_precision(ea_a1, ea_a2),
                "rt": calcular_estadistica_precision(rt_a1, rt_a2)
            },
            "linealidad": linealidad_global.get(elem, None),
            "robustez": robustez_global.get(elem, None)
        }

        # --- Cálculo de recuperación/RPD por matriz de muestras ---
        vol_antes = 49
        vol_muestra = 50
        vol_adicionado = 1
        conc_patron = 1000 if elem in GRUPO_2 else 10

        resultados[elem]["muestras"] = {}
        resultados[elem]["humedad_aplicada_matrices"] = {}
        for matriz_name, datos_m in controles.get("muestras", {}).items():
            res_matriz = []
            len_min = min(len(datos_m["normal"]), len(datos_m["adicionada"]), len(datos_m["duplicada"]))

            for i in range(len_min):
                val_normal = datos_m["normal"][i]["valor"]
                val_adic = datos_m["adicionada"][i]["valor"]
                val_dup = datos_m["duplicada"][i]["valor"]

                # Porcentajes de Recuperación
                numerador_adic = abs(val_adic * (vol_antes + vol_adicionado) - (val_normal * vol_muestra))
                rec_adic = (numerador_adic / (vol_adicionado * conc_patron)) * 100

                numerador_dup = abs(val_dup * (vol_antes + vol_adicionado) - (val_normal * vol_muestra))
                rec_dup = (numerador_dup / (vol_adicionado * conc_patron)) * 100

                # Porcentaje RPD
                prom_adic_dup = (val_adic + val_dup) / 2
                rpd = (abs(val_adic - val_dup) / prom_adic_dup * 100) if prom_adic_dup != 0 else 0

                res_matriz.append({
                    "replica": i + 1,
                    "analista": datos_m["normal"][i]["analista"],
                    "normal": round(val_normal, 4),
                    "adicionada": round(val_adic, 4),
                    "duplicada": round(val_dup, 4),
                    "recuperacion_adic": round(rec_adic, 2),
                    "recuperacion_dup": round(rec_dup, 2),
                    "rpd": round(rpd, 2)
                })

            # Solo agregar al JSON si se encontraron réplicas para esta matriz
            if res_matriz:
                resultados[elem]["muestras"][matriz_name] = res_matriz
                if tipo_analisis == "suelos" and matriz_name in ["arenoso", "arcilloso", "limoso"]:
                    resultados[elem]["humedad_aplicada_matrices"][matriz_name] = obtener_humedad(matriz_name)
                    
    if tipo_analisis == "ras":
        nombres_ras = {"Ca": "Ca soluble", "Mg": "Mg soluble", "Na": "Na soluble", "K": "K soluble"}
        resultados_finales = {}
        for k, v in resultados.items():
            nuevo_nombre = nombres_ras.get(k, k)
            resultados_finales[nuevo_nombre] = v
        return sanear_nan(resultados_finales)

    return sanear_nan(resultados)

@app.post("/api/guardar-reporte")
def guardar_reporte(reporte: ReporteCreate):
    db = SessionLocal()
    try:
        datos = reporte.datos_completos
        
        # Agrupamos los datos crudos de los controles de calidad
        datos_crudos = {
            "mb": datos.get("mb", {}),
            "lcm": datos.get("lcm", {}),
            "ccv": datos.get("ccv", {}),
            "ea": datos.get("ea", {}),
            "rt": datos.get("rt", {})
        }
        
        # Agrupamos los límites calculados y teóricos
        limites = {
            "lod_posible": datos.get("lod_posible"),
            "loq_posible": datos.get("loq_posible"),
            "teorico_lcm": datos.get("teorico_lcm"),
            "teorico_ccv": datos.get("teorico_ccv"),
            "teorico_ea": datos.get("teorico_ea")
        }

        nuevo_reporte = ReporteDB(
            id=str(uuid.uuid4()),
            codigo_informe=reporte.codigo_informe,
            parametro=reporte.parametro,
            matriz=reporte.matriz,
            datos_completos=datos,
            # Mapeo a las columnas explícitas para gráficas y análisis
            datos_crudos=datos_crudos,
            linealidad=datos.get("linealidad"),
            exactitud=datos.get("exactitud"),
            precision=datos.get("precision"),
            robustez=datos.get("robustez"),
            muestras=datos.get("muestras"),
            limites=limites,
            outliers=datos.get("outliers")
        )
        db.add(nuevo_reporte)
        db.commit()
        db.refresh(nuevo_reporte)
        return {"mensaje": "Reporte guardado exitosamente en SQL", "codigo_informe": reporte.codigo_informe}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error al guardar en BD: {str(e)}")
    finally:
        db.close()
        
@app.get("/api/obtener-reporte/{codigo_informe}")
def obtener_reporte(codigo_informe: str):
    db = SessionLocal()
    try:
        # Busca el reporte en la base de datos usando SQLAlchemy
        reporte = db.query(ReporteDB).filter(ReporteDB.codigo_informe == codigo_informe).first()
        
        if not reporte:
            raise HTTPException(status_code=404, detail="Informe no encontrado")
        
        # Retorna los datos completos (el JSON que guardaste originalmente)
        return {
            "codigo_informe": reporte.codigo_informe,
            "parametro": reporte.parametro,
            "matriz": reporte.matriz,
            "fecha_exportacion": reporte.fecha_exportacion,
            "datos_completos": reporte.datos_completos
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()