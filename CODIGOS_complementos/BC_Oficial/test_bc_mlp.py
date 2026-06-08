"""
test_bc_mlp.py  —  Behavior Cloning con MLP (PyTorch)
======================================================
Carga modelo_bc.pt y ejecuta el brazo robótico usando la red neuronal
para predecir los pasos exactos desde la posición del trozo detectado.

Flujo por ciclo:
  1. HOME (C=400, H=400)
  2. Gripper cierra (PINZA 0)
  3. Cámara: malla + círculos → ESPACIO para capturar
  4. MLP predice (base, codo, hombro) desde (cx_norm, cy_norm)
  5. Gripper abre (PINZA 90)
  6. Ejecuta movimientos predichos: Base → Codo → Hombro
  7. Gripper cierra (PINZA 0) — agarre
  8. HOME: Hombro → Codo → Base
  9. Gripper abre (PINZA 90) — soltar
 10. Gripper cierra (PINZA 0) — listo para siguiente
"""

import os, json, time, argparse
import numpy as np
import cv2
import torch 
import torch.nn as nn

# ─────────────────────────────────────────────
#  CONFIGURACION
# ─────────────────────────────────────────────
MODELO_PT  = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_NUEVO\modelo_bc.pt"
CALIB_JSON = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_NUEVO\calibracion_cuadricula.json"

SERIAL_PORT  = "COM3"
SERIAL_BAUD  = 115200
CAMERA_INDEX = 3

HOME_POSITION = {"base":0,"hombro":400,"codo":400,"muneca":0,"rotacion":0}
JOINT_LIMITS  = {ax:(-3200,3200) for ax in HOME_POSITION}
AXIS_CMD      = {"base":"BASE","hombro":"HOMBRO","codo":"CODO",
                 "muneca":"GRIPPER","rotacion":"GIRO"}
PINZA_MIN = 0
PINZA_MAX = 90
GRID_COLS = 12
GRID_ROWS = 9

YOLO_CONF = 0.20
YOLO_CLASSES = [
    "apple","pear","peach","plum","apricot","cherry","strawberry","raspberry",
    "blueberry","blackberry","grape","watermelon","melon","cantaloupe","honeydew",
    "banana","mango","papaya","pineapple chunk","kiwi","orange slice","mandarin",
    "lemon slice","lime slice","fig","date","lychee","guava","passion fruit",
    "dragon fruit","star fruit","persimmon","pomegranate seed","tomato",
    "cherry tomato","carrot piece","broccoli floret","cauliflower","lettuce piece",
    "cucumber slice","zucchini","eggplant","bell pepper","corn kernel","pea",
    "green bean","asparagus","artichoke","celery piece","beet","radish","turnip",
    "potato chunk","sweet potato","mushroom","onion piece","leek","spinach","kale",
    "cabbage piece","brussels sprout","bok choy","chicken piece","beef piece",
    "pork piece","lamb piece","turkey piece","sausage slice","meatball","nugget",
    "shrimp","fish piece","salmon chunk","tuna piece","squid piece","octopus piece",
    "crab meat","boiled egg","fried egg piece","omelette piece","tofu cube",
    "tempeh piece","pasta piece","noodle","gnocchi","dumpling","rice ball",
    "bread piece","crouton","cheese cube","mozzarella","ham piece","olive",
    "pickle slice","sun-dried tomato","chickpea","lentil","bean",
    "food piece","fruit piece","vegetable piece","meat piece",
]
COLORES = [
    (0,255,0),(255,100,0),(0,100,255),(255,0,255),(0,255,255),
    (255,255,0),(100,255,100),(255,150,50),(50,200,255),(200,50,255),
]


# ─────────────────────────────────────────────
#  MLP — misma arquitectura que train_mlp.py
# ─────────────────────────────────────────────
class BrazoMLP(nn.Module):
    def __init__(self, hidden, dropout=0.05):
        super().__init__()
        capas = []
        in_dim = 2
        for h in hidden:
            capas += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        capas.append(nn.Linear(in_dim, 3))
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x)


# ─────────────────────────────────────────────
#  PREDICTOR MLP
# ─────────────────────────────────────────────
class MLPPredictor:
    """
    Carga modelo_bc.pt y predice (base, codo, hombro) en pasos
    a partir de (cx_norm, cy_norm).
    """

    def __init__(self, pt_path):
        print(f"Cargando modelo MLP: {pt_path}")
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

        self.y_mean = np.array(ckpt["y_mean"], dtype=np.float32)
        self.y_std  = np.array(ckpt["y_std"],  dtype=np.float32)
        self.output_names = ckpt.get("output_names", ["base","codo","hombro"])
        trained_at = ckpt.get("trained_at", "?")
        n_params   = ckpt.get("n_params", "?")

        hidden  = ckpt["hidden"]
        dropout = ckpt.get("dropout", 0.05)
        self.model = BrazoMLP(hidden, dropout)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        print(f"  Arq.       : 2 -> {' -> '.join(str(h) for h in hidden)} -> 3")
        print(f"  Parámetros : {n_params}")
        print(f"  Entrenado  : {trained_at}")
        print(f"  y_mean     : {self.y_mean.tolist()}")
        print(f"  y_std      : {self.y_std.tolist()}\n")

    def predecir(self, cx_norm, cy_norm):
        """
        Devuelve dict con pasos netos predichos para cada articulación.
        Los valores se redondean al entero más cercano.
        """
        x = torch.tensor([[cx_norm, cy_norm]], dtype=torch.float32)
        with torch.no_grad():
            y_norm = self.model(x).numpy()[0]
        y_raw  = y_norm * self.y_std + self.y_mean
        y_int  = {name: int(round(float(v)))
                  for name, v in zip(self.output_names, y_raw)}
        return y_int, y_raw

    def pasos_a_steps(self, prediccion):
        """
        Convierte el dict de predicción en lista de steps compatibles
        con RobotInterface.ejecutar_paso().
        Orden: Base → Codo → Hombro → GRIP 0
        """
        steps = []
        orden = [("base","B"), ("codo","C"), ("hombro","H")]
        for nombre, letra in orden:
            val = prediccion.get(nombre, 0)
            if val == 0:
                continue
            cmd  = f"{letra}+" if val > 0 else f"{letra}-"
            steps.append({
                "cmd":            cmd,
                "steps_executed": abs(val),
                "action":         None,
                "angulo_pinza":   None,
            })
        steps.append({"cmd":"GRIP 0","steps_executed":0,"action":10,"angulo_pinza":0})
        return steps


# ─────────────────────────────────────────────
#  CUADRICULA  (malla + esquinas + zona activa)
# ─────────────────────────────────────────────
class Cuadricula:
    COLOR_LINEA   = (0,255,255)
    COLOR_NUMERO  = (255,255,0)
    COLOR_RELLENO = (0,60,60)

    def __init__(self, json_path):
        self.cols=GRID_COLS; self.rows=GRID_ROWS
        self.M=None; self.M_inv=None
        try:
            with open(json_path,"r") as f: d=json.load(f)
            self.cols=d["grid_cols"]; self.rows=d["grid_rows"]
            src=np.float32([[0,0],[self.cols,0],[self.cols,self.rows],[0,self.rows]])
            dst=np.float32([tuple(p) for p in d["puntos"]])
            self.M=cv2.getPerspectiveTransform(src,dst)
            self.M_inv=np.linalg.inv(self.M)
            print(f"[Cuadricula] {self.cols}x{self.rows} | {(self.cols+1)*(self.rows+1)} esquinas\n")
        except Exception as e:
            print(f"[Cuadricula] ERROR: {e}\n")

    @property
    def disponible(self): return self.M is not None

    def _t(self,pts):
        return cv2.perspectiveTransform(np.float32(pts).reshape(-1,1,2),self.M).reshape(-1,2)

    def poligono_px(self):
        if not self.disponible: return None
        return self._t([[0,0],[self.cols,0],[self.cols,self.rows],[0,self.rows]]).astype(np.int32)

    def punto_dentro(self,px,py):
        poly=self.poligono_px()
        if poly is None: return True
        return cv2.pointPolygonTest(poly,(float(px),float(py)),False) >= 0

    def dibujar_zona_activa(self,frame):
        poly=self.poligono_px()
        if poly is None: return frame
        mask=np.zeros(frame.shape[:2],dtype=np.uint8)
        cv2.fillPoly(mask,[poly],255)
        dark=frame.copy()
        dark[mask==0]=(dark[mask==0]*0.30).astype(np.uint8)
        cv2.addWeighted(dark,1.0,frame,0.0,0,frame)
        cv2.polylines(frame,[poly],True,(0,255,100),3)
        return frame

    def info_celda(self,px,py):
        if not self.disponible: return {"celda":None,"fila":None,"columna":None}
        gx,gy=cv2.perspectiveTransform(np.float32([[[px,py]]]),self.M_inv)[0][0]
        if 0<=gx<=self.cols and 0<=gy<=self.rows:
            col=min(int(gx),self.cols-1); row=min(int(gy),self.rows-1)
            return {"celda":row*self.cols+col+1,"fila":row+1,"columna":col+1}
        return {"celda":None,"fila":None,"columna":None}

    def _esq_num(self,col_e,row_e): return row_e*(self.cols+1)+col_e+1

    def esquina_mas_cercana(self,px,py):
        if not self.disponible: return None
        gx,gy=cv2.perspectiveTransform(np.float32([[[px,py]]]),self.M_inv)[0][0]
        col_e=int(round(np.clip(gx,0,self.cols)))
        row_e=int(round(np.clip(gy,0,self.rows)))
        num=self._esq_num(col_e,row_e)
        pt=self._t([[col_e,row_e]])[0]
        return {"esquina":num,"col_e":col_e,"row_e":row_e,
                "px_img":float(pt[0]),"py_img":float(pt[1])}

    def dibujar(self,frame,highlight_celda=None,prediccion_px=None):
        """
        highlight_celda : celda del trozo detectado (naranja)
        prediccion_px   : (px, py) donde apuntará el brazo según MLP (cruz magenta)
        """
        if not self.disponible: return frame

        # Rellenos
        ov=frame.copy()
        for r in range(self.rows):
            for c in range(self.cols):
                num=r*self.cols+c+1
                pts=self._t([[c,r],[c+1,r],[c+1,r+1],[c,r+1]]).astype(np.int32)
                col=(0,80,160) if num==highlight_celda else self.COLOR_RELLENO
                cv2.fillPoly(ov,[pts],col)
        cv2.addWeighted(ov,0.35,frame,0.65,0,frame)

        if highlight_celda and 1<=highlight_celda<=self.cols*self.rows:
            r=(highlight_celda-1)//self.cols; c=(highlight_celda-1)%self.cols
            pts=self._t([[c,r],[c+1,r],[c+1,r+1],[c,r+1]]).astype(np.int32)
            cv2.fillPoly(frame,[pts],(0,140,255))

        # Líneas
        for c in range(self.cols+1):
            p1=self._t([[c,0]]).astype(int)[0]; p2=self._t([[c,self.rows]]).astype(int)[0]
            cv2.line(frame,tuple(p1),tuple(p2),self.COLOR_LINEA,2 if c in (0,self.cols) else 1)
        for r in range(self.rows+1):
            p1=self._t([[0,r]]).astype(int)[0]; p2=self._t([[self.cols,r]]).astype(int)[0]
            cv2.line(frame,tuple(p1),tuple(p2),self.COLOR_LINEA,2 if r in (0,self.rows) else 1)

        # Números
        for r in range(self.rows):
            for c in range(self.cols):
                num=r*self.cols+c+1
                cx=self._t([[c+0.5,r+0.5]])[0].astype(int)
                txt=str(num)
                (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.36,1)
                col=(255,255,255) if num==highlight_celda else self.COLOR_NUMERO
                cv2.putText(frame,txt,(int(cx[0])-tw//2,int(cx[1])+th//2),
                            cv2.FONT_HERSHEY_SIMPLEX,0.36,col,1)

        # Círculos en esquinas
        R=9
        for row_e in range(self.rows+1):
            for col_e in range(self.cols+1):
                pt=self._t([[col_e,row_e]])[0].astype(int)
                cv2.circle(frame,tuple(pt),R,(0,0,180),2)
                cv2.line(frame,(pt[0]-4,pt[1]),(pt[0]+4,pt[1]),(0,180,180),1)
                cv2.line(frame,(pt[0],pt[1]-4),(pt[0],pt[1]+4),(0,180,180),1)

        return frame


# ─────────────────────────────────────────────
#  ROBOT INTERFACE
# ─────────────────────────────────────────────
class RobotInterface:
    def __init__(self,simulate=False):
        self.simulate=simulate
        self._positions=dict(HOME_POSITION)
        self._angulo_pinza=0; self._ser=None

    def __enter__(self):
        if not self.simulate:
            import serial
            self._ser=serial.Serial(SERIAL_PORT,SERIAL_BAUD,timeout=10)
            time.sleep(2); self._ser.flushInput()
            deadline=time.time()+20
            while time.time()<deadline:
                if self._ser.readline().decode(errors="ignore").strip()=="READY": break
            print(f"Arduino {SERIAL_PORT} @ {SERIAL_BAUD}.")
        else:
            print("Robot [SIMULADO].")
        return self

    def __exit__(self,*_):
        if self._ser and self._ser.is_open: self._ser.close()
        print("Robot desconectado.")

    def _send(self,cmd):
        if self.simulate:
            print(f"    [SIM] -> {cmd}"); time.sleep(0.1); return "OK"
        self._ser.flushInput(); self._ser.write((cmd+"\n").encode())
        deadline=time.time()+20
        while time.time()<deadline:
            l=self._ser.readline().decode(errors="ignore").strip()
            if l=="OK": return "OK"
            if l=="ERR": return "ERR"
        return "TIMEOUT"

    def set_gripper(self,angulo,label=""):
        angulo=int(np.clip(angulo,PINZA_MIN,PINZA_MAX))
        print(f"  Gripper -> {angulo}deg{' ('+label+')' if label else ''}")
        resp=self._send(f"PINZA {angulo}")
        if resp=="OK": self._angulo_pinza=angulo
        return resp

    def go_home(self):
        print("  HOME (Hombro -> Codo -> Base)...")
        for ax in ["hombro","codo","muneca","rotacion","base"]:
            tgt=HOME_POSITION[ax]; cur=self._positions[ax]; diff=tgt-cur
            if diff==0: continue
            print(f"    {ax}: {cur:+d} -> {tgt:+d} ({diff:+d})")
            self._send(f"{AXIS_CMD[ax]} {diff}"); self._positions[ax]=tgt
        print("  HOME OK.")

    def ejecutar_paso(self,step):
        cmd=step.get("cmd",""); pasos=step.get("steps_executed",0)
        mapa={"B+":("base",+1),"B-":("base",-1),
              "C+":("codo",+1),"C-":("codo",-1),
              "H+":("hombro",+1),"H-":("hombro",-1)}
        if cmd=="GRIP 0":  return self.set_gripper(0,"agarrar")
        if cmd=="GRIP 90": return self.set_gripper(90,"abrir")
        if cmd not in mapa: print(f"    [SKIP] {cmd}"); return "SKIP"
        ax,d=mapa[cmd]
        new=int(np.clip(self._positions[ax]+d*pasos,*JOINT_LIMITS[ax]))
        diff=new-self._positions[ax]
        if diff==0: return "OK"
        print(f"    {cmd} {abs(diff):>6}  ({ax}: {self._positions[ax]:+d} -> {new:+d})")
        resp=self._send(f"{AXIS_CMD[ax]} {diff}")
        if resp=="OK": self._positions[ax]=new
        return resp


# ─────────────────────────────────────────────
#  VISION
# ─────────────────────────────────────────────
class VisionPipeline:
    def __init__(self,cuadricula,simulate=False):
        self.simulate=simulate; self.cuadricula=cuadricula
        self._cap=None; self._yolo=None

    def __enter__(self):
        if not self.simulate:
            from ultralytics import YOLOWorld
            self._cap=cv2.VideoCapture(CAMERA_INDEX)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
            if not self._cap.isOpened(): raise RuntimeError(f"No camara {CAMERA_INDEX}")
            print(f"Camara {CAMERA_INDEX} abierta.")
            print("Cargando YOLOWorld...")
            self._yolo=YOLOWorld("yolov8m-world.pt")
            self._yolo.set_classes(YOLO_CLASSES)
            print("YOLOWorld listo.\n")
        else:
            print("Vision [SIMULADA].")
        return self

    def __exit__(self,*_):
        if self._cap: self._cap.release()

    def read_frame(self):
        if self.simulate or self._cap is None: return None
        ret,frame=self._cap.read(); return frame if ret else None

    def detectar(self,frame,predictor=None):
        """
        Detecta trozos DENTRO de la malla.
        Si predictor != None, muestra en tiempo real la prediccion MLP.
        """
        if self.simulate or frame is None:
            return [],np.zeros((480,640,3),dtype=np.uint8)

        h,w=frame.shape[:2]
        results=self._yolo.predict(frame,conf=YOLO_CONF,iou=0.25,verbose=False)[0]
        detections=[]
        for box in results.boxes:
            x1,y1,x2,y2=box.xyxy[0].tolist()
            cls=int(box.cls[0]); cx_px=(x1+x2)/2; cy_px=(y1+y2)/2
            # Filtro: solo dentro de la malla
            if not self.cuadricula.punto_dentro(cx_px,cy_px):
                continue
            ci=self.cuadricula.info_celda(cx_px,cy_px)
            detections.append({
                "bbox":(x1,y1,x2,y2),"cx_px":cx_px,"cy_px":cy_px,
                "cx_norm":cx_px/w,"cy_norm":cy_px/h,
                "conf":float(box.conf[0]),"color":COLORES[cls%len(COLORES)],
                "celda_info":ci,
            })
        detections.sort(key=lambda x:x["conf"],reverse=True)

        # Construir frame anotado
        annotated=frame.copy()
        annotated=self.cuadricula.dibujar_zona_activa(annotated)

        hl_c=detections[0]["celda_info"]["celda"] if detections else None
        annotated=self.cuadricula.dibujar(annotated,highlight_celda=hl_c)

        # Bounding boxes
        for det in detections:
            x1,y1,x2,y2=[int(v) for v in det["bbox"]]
            ci=det["celda_info"]; color=det["color"]
            c_s=f"C{ci['celda']}" if ci["celda"] else "fuera"
            tag=f"{c_s}  {det['conf']:.2f}"
            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
            cx_i,cy_i=int(det["cx_px"]),int(det["cy_px"])
            cv2.circle(annotated,(cx_i,cy_i),6,(0,0,255),-1)
            cv2.circle(annotated,(cx_i,cy_i),6,(255,255,255),1)
            (tw,th),_=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
            cv2.rectangle(annotated,(x1,y1-th-8),(x1+tw+4,y1),color,-1)
            cv2.putText(annotated,tag,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1)

        # Objetivo principal + predicción MLP en tiempo real
        if detections:
            best=detections[0]
            x1,y1,x2,y2=[int(v) for v in best["bbox"]]
            cv2.rectangle(annotated,(x1-3,y1-3),(x2+3,y2+3),(0,255,255),3)
            ci=best["celda_info"]

            if predictor:
                pred,pred_raw=predictor.predecir(best["cx_norm"],best["cy_norm"])
                txt_pred=(f"MLP -> B:{pred['base']:+d}  C:{pred['codo']:+d}  H:{pred['hombro']:+d}")
                cv2.putText(annotated,txt_pred,(x1,y2+26),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
                # Texto de celda
                txt_obj=f"Celda {ci['celda']}  cx={best['cx_norm']:.3f}  cy={best['cy_norm']:.3f}"
                cv2.putText(annotated,txt_obj,(x1,y2+52),
                            cv2.FONT_HERSHEY_SIMPLEX,0.48,(180,255,180),1)
            else:
                txt=(f"OBJETIVO  Celda {ci['celda']}"
                     if ci["celda"] else "OBJETIVO (fuera)")
                cv2.putText(annotated,txt,(x1,y2+26),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        # Leyenda
        cv2.rectangle(annotated,(0,h-44),(w,h),(0,0,0),-1)
        cv2.putText(annotated,
            f"MLP BC  |  Trozos: {len(detections)} (solo dentro malla)  |  "
            "ESPACIO=confirmar  |  ESC/q=salir",
            (8,h-24),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,255,255),1)
        cv2.putText(annotated,
            "Naranja=objetivo  |  Zona oscura=ignorada  |  "
            "Prediccion MLP en tiempo real (cyan)",
            (8,h-6),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)

        return detections,annotated


# ─────────────────────────────────────────────
#  CAPTURA INTERACTIVA
# ─────────────────────────────────────────────
def capturar_objetivo(vision, predictor):
    """
    Muestra la predicción MLP en tiempo real.
    ESPACIO confirma el trozo actual y devuelve el resultado.
    """
    VENTANA="MLP BC  |  ESPACIO=confirmar  |  ESC/q=salir"
    cv2.namedWindow(VENTANA,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA,1100,680); cv2.moveWindow(VENTANA,50,30)
    try: cv2.setWindowProperty(VENTANA,cv2.WND_PROP_TOPMOST,1)
    except: pass

    espera=np.zeros((680,1100,3),dtype=np.uint8)
    cv2.putText(espera,"Iniciando camara...",(290,300),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),2)
    cv2.putText(espera,"La prediccion MLP aparece en tiempo real (cyan)",(200,370),
                cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),1)
    cv2.putText(espera,"ESPACIO=confirmar   ESC/q=salir",(320,430),
                cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),1)
    cv2.imshow(VENTANA,espera); cv2.waitKey(1)

    for _ in range(15):
        f=vision.read_frame()
        if f is not None: cv2.imshow(VENTANA,f); cv2.waitKey(1); break
        cv2.waitKey(50)

    resultado=None
    while True:
        frame=vision.read_frame()
        if frame is None:
            cv2.imshow(VENTANA,espera)
            if cv2.waitKey(80)&0xFF in (27,ord('q')): resultado="SALIR"; break
            continue

        dets,annotated=vision.detectar(frame, predictor=predictor)
        cv2.imshow(VENTANA,annotated)
        key=cv2.waitKey(30)&0xFF

        if key in (27,ord('q')): resultado="SALIR"; break
        if key==32:
            if not dets: print("  Sin deteccion YOLO."); continue
            best=dets[0]; ci=best["celda_info"]
            if ci["celda"] is None: print("  Fuera de malla."); continue
            pred,pred_raw=predictor.predecir(best["cx_norm"],best["cy_norm"])
            resultado={
                "celda_info":ci,
                "cx_norm":best["cx_norm"],
                "cy_norm":best["cy_norm"],
                "conf":best["conf"],
                "prediccion":pred,
                "prediccion_raw":pred_raw,
            }
            print(f"  Confirmado: Celda {ci['celda']}  "
                  f"cx={best['cx_norm']:.4f}  cy={best['cy_norm']:.4f}")
            print(f"  MLP predice: B={pred['base']:+d}  C={pred['codo']:+d}  H={pred['hombro']:+d}")
            break

    cv2.destroyWindow(VENTANA); cv2.waitKey(1)
    return resultado


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main(simulate=False):
    # Cargar recursos
    cuadricula = Cuadricula(CALIB_JSON)
    predictor  = MLPPredictor(MODELO_PT)
    ciclo      = 0

    with RobotInterface(simulate=simulate) as robot:
        vision = VisionPipeline(cuadricula=cuadricula, simulate=simulate)
        with vision:
            print("=" * 62)
            print("  MLP BEHAVIOR CLONING TEST")
            print("  Red neuronal predice pasos desde posicion del trozo")
            print("=" * 62)
            print()
            print("  La prediccion MLP se muestra en TIEMPO REAL en la ventana")
            print("  mientras apuntas la camara al plato.")
            print()
            print("  ESPACIO = confirmar y ejecutar  |  ESC/q = salir")
            print()

            continuar=True
            while continuar:
                ciclo+=1
                print(f"\n{'='*62}  CICLO {ciclo}")

                print("\n[1] HOME..."); robot.go_home()
                print("\n[2] Cerrando gripper..."); robot.set_gripper(PINZA_MIN,"captura"); time.sleep(0.4)

                print("\n[3] Abre camara — apunta al plato y presiona ESPACIO...")
                captura=capturar_objetivo(vision, predictor)
                if captura is None or captura=="SALIR":
                    print("  Saliendo..."); continuar=False; break

                ci   = captura["celda_info"]
                pred = captura["prediccion"]

                print(f"\n[4] MLP prediccion final:")
                print(f"  Entrada  : cx={captura['cx_norm']:.4f}  cy={captura['cy_norm']:.4f}")
                print(f"  Celda    : {ci['celda']} (Fila {ci['fila']}, Col {ci['columna']})")
                print(f"  Base     : {pred['base']:+d} pasos")
                print(f"  Codo     : {pred['codo']:+d} pasos")
                print(f"  Hombro   : {pred['hombro']:+d} pasos")

                # Convertir prediccion a steps ejecutables
                steps = predictor.pasos_a_steps(pred)

                print(f"\n[5] Abriendo gripper..."); robot.set_gripper(PINZA_MAX,"pre-agarre"); time.sleep(0.5)

                print(f"\n[6] Ejecutando movimientos MLP ({len(steps)-1} ejes)...")
                pasos_mov=[s for s in steps if s.get("cmd")!="GRIP 0"]
                for i,step in enumerate(pasos_mov,1):
                    print(f"  [{i}/{len(pasos_mov)}] ",end="")
                    resp=robot.ejecutar_paso(step)
                    if resp not in ("OK","SKIP"): print(f"  WARN: {resp}")
                    time.sleep(0.05)

                print(f"\n[7] Cerrando gripper — AGARRE..."); robot.set_gripper(PINZA_MIN,"agarrando"); time.sleep(0.8)
                print(f"\n[8] HOME..."); robot.go_home(); time.sleep(0.3)
                print(f"\n[9] Abriendo — SOLTANDO..."); robot.set_gripper(PINZA_MAX,"soltando"); time.sleep(0.8)
                print(f"\n[10] Cerrando para siguiente..."); robot.set_gripper(PINZA_MIN,"listo"); time.sleep(0.4)

                print(f"\n  Ciclo {ciclo} OK.")
                print("  ENTER=siguiente | q=salir: ",end="",flush=True)
                try:
                    if input().strip().lower()=="q": continuar=False
                except (EOFError,KeyboardInterrupt):
                    continuar=False

    cv2.destroyAllWindows()
    print("\nPrograma terminado.")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="MLP BC Test — Brazo Robótico")
    parser.add_argument("--sim",    action="store_true", help="Simulacion sin hardware")
    parser.add_argument("--modelo", default=MODELO_PT)
    parser.add_argument("--calib",  default=CALIB_JSON)
    args=parser.parse_args()
    if args.modelo!=MODELO_PT: globals()["MODELO_PT"]=args.modelo
    if args.calib !=CALIB_JSON: globals()["CALIB_JSON"]=args.calib
    main(simulate=args.sim)
