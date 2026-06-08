"""
test_bc.py  —  Behavior Cloning con malla + esquinas
=====================================================
Usa demonstrations_clean.pkl (celdas + corners) para ejecutar
el brazo imitando los movimientos de la demo más cercana.

Flujo por ciclo:
  1. HOME (C=400, H=400)
  2. Gripper cierra (PINZA 0)
  3. Cámara: malla + círculos en esquinas → ESPACIO para capturar
  4. Cámara cierra
  5. BC lookup: busca mejor demo por celda Y esquina más cercana
  6. Gripper abre (PINZA 90)
  7. Reproduce pasos: Base → Codo → Hombro
  8. Gripper cierra (PINZA 0) — agarre
  9. Vuelve a HOME: Hombro → Codo → Base
 10. Gripper abre (PINZA 90) — soltar
 11. Gripper cierra (PINZA 0) — listo para siguiente
 12. ENTER = siguiente ciclo  |  q = salir
"""

import os, json, pickle, time, argparse
import numpy as np
import cv2
from collections import Counter

CLEAN_PKL  = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_NUEVO\demos\demonstrations_clean.pkl"
CALIB_JSON = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_NUEVO\calibracion_cuadricula.json"

SERIAL_PORT  = "COM5"
SERIAL_BAUD  = 115200
CAMERA_INDEX = 2

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


# ─── DATASET BC ───────────────────────────────────────────────────────────────
class BCDataset:
    def __init__(self, pkl_path):
        print(f"Cargando dataset: {pkl_path}")
        with open(pkl_path,"rb") as f:
            data = pickle.load(f)
        self.episodes = data.get("episodes",[])
        print(f"  {len(self.episodes)} demos cargadas.")

        self._idx_celda   = {}
        self._idx_esquina = {}
        for ep in self.episodes:
            c = ep.get("celda_objetivo")
            e = ep.get("esquina_objetivo")
            if c is not None: self._idx_celda.setdefault(c,[]).append(ep)
            if e is not None: self._idx_esquina.setdefault(e,[]).append(ep)

        n_solo  = sum(1 for ep in self.episodes if ep.get("celda_objetivo") and not ep.get("esquina_objetivo"))
        n_esq   = sum(1 for ep in self.episodes if ep.get("esquina_objetivo"))
        print(f"  Demos celda: {n_solo}  |  Demos esquina: {n_esq}")
        print(f"  Celdas: {len(self._idx_celda)}  |  Esquinas: {len(self._idx_esquina)}\n")

    @staticmethod
    def _celda_rc(celda):
        return (celda-1)//GRID_COLS, (celda-1)%GRID_COLS

    @staticmethod
    def _esq_rc(esquina):
        return (esquina-1)//(GRID_COLS+1), (esquina-1)%(GRID_COLS+1)

    def _mejor(self, lista):
        ex = [d for d in lista if d.get("success")]
        return ex[0] if ex else lista[0]

    def buscar_demo(self, celda_target, esquina_target=None):
        # 1. Esquina exacta
        if esquina_target and esquina_target in self._idx_esquina:
            d = self._mejor(self._idx_esquina[esquina_target])
            return d,"esquina_exacta",d.get("celda_objetivo"),esquina_target,0.0
        # 2. Celda exacta
        if celda_target in self._idx_celda:
            d = self._mejor(self._idx_celda[celda_target])
            return d,"celda_exacta",celda_target,d.get("esquina_objetivo"),0.0
        # 3. Esquina mas cercana
        if esquina_target:
            tr,tc = self._esq_rc(esquina_target)
            bd,be = float("inf"),None
            for e in self._idx_esquina:
                r,c = self._esq_rc(e)
                dist = ((r-tr)**2+(c-tc)**2)**0.5
                if dist<bd: bd,be = dist,e
            if be is not None:
                d = self._mejor(self._idx_esquina[be])
                return d,"esquina_cercana",d.get("celda_objetivo"),be,bd
        # 4. Celda mas cercana
        tr,tc = self._celda_rc(celda_target)
        bd,bc = float("inf"),None
        for c in self._idx_celda:
            r,cc = self._celda_rc(c)
            dist = ((r-tr)**2+(cc-tc)**2)**0.5
            if dist<bd: bd,bc = dist,c
        d = self._mejor(self._idx_celda[bc])
        return d,"celda_cercana",bc,d.get("esquina_objetivo"),bd

    def celdas_con_demo(self):   return set(self._idx_celda.keys())
    def esquinas_con_demo(self): return {e:len(eps) for e,eps in self._idx_esquina.items()}


# ─── CUADRICULA ───────────────────────────────────────────────────────────────
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
            print(f"[Cuadricula] {self.cols}x{self.rows} celdas | {(self.cols+1)*(self.rows+1)} esquinas\n")
        except Exception as e:
            print(f"[Cuadricula] ERROR: {e}\n")

    @property
    def disponible(self): return self.M is not None

    def _t(self,pts):
        return cv2.perspectiveTransform(np.float32(pts).reshape(-1,1,2),self.M).reshape(-1,2)

    def info_celda(self,px,py):
        if not self.disponible: return {"celda":None,"fila":None,"columna":None}
        gx,gy = cv2.perspectiveTransform(np.float32([[[px,py]]]),self.M_inv)[0][0]
        if 0<=gx<=self.cols and 0<=gy<=self.rows:
            col=min(int(gx),self.cols-1); row=min(int(gy),self.rows-1)
            return {"celda":row*self.cols+col+1,"fila":row+1,"columna":col+1}
        return {"celda":None,"fila":None,"columna":None}

    def _esq_num(self,col_e,row_e): return row_e*(self.cols+1)+col_e+1

    def poligono_px(self):
        """Devuelve los 4 vértices del borde externo de la malla en píxeles (int32)."""
        if not self.disponible:
            return None
        pts = self._t([[0,0],[self.cols,0],[self.cols,self.rows],[0,self.rows]])
        return pts.astype(np.int32)

    def punto_dentro(self, px, py):
        """True si el punto (px, py) está dentro del polígono de la malla."""
        poly = self.poligono_px()
        if poly is None:
            return True   # si no hay malla, aceptar todo
        return cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0

    def dibujar_zona_activa(self, frame):
        """Oscurece el área FUERA de la malla para indicar la zona de detección."""
        poly = self.poligono_px()
        if poly is None:
            return frame
        # Crear máscara: 255 dentro, 0 fuera
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        # Oscurecer zona exterior
        dark = frame.copy()
        dark[mask == 0] = (dark[mask == 0] * 0.30).astype(np.uint8)
        cv2.addWeighted(dark, 1.0, frame, 0.0, 0, frame)
        # Borde brillante de la zona activa
        cv2.polylines(frame, [poly], True, (0, 255, 100), 3)
        return frame

    def esquina_mas_cercana(self,px,py):
        if not self.disponible: return None
        gx,gy = cv2.perspectiveTransform(np.float32([[[px,py]]]),self.M_inv)[0][0]
        col_e=int(round(np.clip(gx,0,self.cols)))
        row_e=int(round(np.clip(gy,0,self.rows)))
        num=self._esq_num(col_e,row_e)
        pt=self._t([[col_e,row_e]])[0]
        return {"esquina":num,"col_e":col_e,"row_e":row_e,
                "px_img":float(pt[0]),"py_img":float(pt[1])}

    def dibujar(self,frame,highlight_celda=None,celdas_con_demo=None,
                esquinas_con_demo=None,esquina_highlight=None):
        if not self.disponible: return frame
        if celdas_con_demo   is None: celdas_con_demo   = set()
        if esquinas_con_demo is None: esquinas_con_demo  = {}

        # Rellenos de celda
        ov = frame.copy()
        for r in range(self.rows):
            for c in range(self.cols):
                num  = r*self.cols+c+1
                pts  = self._t([[c,r],[c+1,r],[c+1,r+1],[c,r+1]]).astype(np.int32)
                if num==highlight_celda:          col=(0,80,160)
                elif num in celdas_con_demo:      col=(0,70,0)
                else:                             col=self.COLOR_RELLENO
                cv2.fillPoly(ov,[pts],col)
        cv2.addWeighted(ov,0.35,frame,0.65,0,frame)

        if highlight_celda and 1<=highlight_celda<=self.cols*self.rows:
            r=(highlight_celda-1)//self.cols; c=(highlight_celda-1)%self.cols
            pts=self._t([[c,r],[c+1,r],[c+1,r+1],[c,r+1]]).astype(np.int32)
            cv2.fillPoly(frame,[pts],(0,140,255))

        for cn in celdas_con_demo:
            if cn==highlight_celda: continue
            r=(cn-1)//self.cols; c=(cn-1)%self.cols
            pts=self._t([[c,r],[c+1,r],[c+1,r+1],[c,r+1]]).astype(np.int32)
            cv2.polylines(frame,[pts],True,(0,200,0),2)

        # Lineas
        for c in range(self.cols+1):
            p1=self._t([[c,0]]).astype(int)[0]; p2=self._t([[c,self.rows]]).astype(int)[0]
            cv2.line(frame,tuple(p1),tuple(p2),self.COLOR_LINEA,2 if c in (0,self.cols) else 1)
        for r in range(self.rows+1):
            p1=self._t([[0,r]]).astype(int)[0]; p2=self._t([[self.cols,r]]).astype(int)[0]
            cv2.line(frame,tuple(p1),tuple(p2),self.COLOR_LINEA,2 if r in (0,self.rows) else 1)

        # Numeros
        for r in range(self.rows):
            for c in range(self.cols):
                num=r*self.cols+c+1
                cx=self._t([[c+0.5,r+0.5]])[0].astype(int)
                txt=str(num)
                (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.36,1)
                col=(255,255,255) if num==highlight_celda else self.COLOR_NUMERO
                cv2.putText(frame,txt,(int(cx[0])-tw//2,int(cx[1])+th//2),
                            cv2.FONT_HERSHEY_SIMPLEX,0.36,col,1)

        # Circulos en esquinas
        R=11
        for row_e in range(self.rows+1):
            for col_e in range(self.cols+1):
                num_e = self._esq_num(col_e,row_e)
                pt    = self._t([[col_e,row_e]])[0].astype(int)
                ctr   = tuple(pt)
                cnt   = esquinas_con_demo.get(num_e,0)

                if num_e==esquina_highlight:
                    cv2.circle(frame,ctr,R+4,(255,255,255),2)
                    cv2.circle(frame,ctr,R,(0,255,255),-1)
                    cv2.circle(frame,ctr,R,(255,255,255),1)
                elif cnt>0:
                    cv2.circle(frame,ctr,R,(0,0,220),-1)
                    cv2.circle(frame,ctr,R,(255,255,255),1)
                    t=str(cnt)
                    (tw2,th2),_=cv2.getTextSize(t,cv2.FONT_HERSHEY_SIMPLEX,0.32,1)
                    cv2.putText(frame,t,(ctr[0]-tw2//2,ctr[1]+th2//2),
                                cv2.FONT_HERSHEY_SIMPLEX,0.32,(255,255,255),1)
                else:
                    cv2.circle(frame,ctr,R,(0,0,200),2)
                    cv2.line(frame,(ctr[0]-4,ctr[1]),(ctr[0]+4,ctr[1]),(0,200,200),1)
                    cv2.line(frame,(ctr[0],ctr[1]-4),(ctr[0],ctr[1]+4),(0,200,200),1)
        return frame


# ─── ROBOT ────────────────────────────────────────────────────────────────────
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
        estado="cerrada" if angulo==0 else f"abierta {angulo}deg"
        print(f"  Gripper -> {angulo}deg ({estado}){' -- '+label if label else ''}")
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


# ─── VISION ───────────────────────────────────────────────────────────────────
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

    def detectar(self,frame,celdas_con_demo=None,esquinas_con_demo=None):
        if celdas_con_demo   is None: celdas_con_demo   = set()
        if esquinas_con_demo is None: esquinas_con_demo  = {}
        if self.simulate or frame is None:
            return [],np.zeros((480,640,3),dtype=np.uint8)

        h,w=frame.shape[:2]
        results=self._yolo.predict(frame,conf=YOLO_CONF,iou=0.25,verbose=False)[0]
        detections=[]
        for box in results.boxes:
            x1,y1,x2,y2=box.xyxy[0].tolist()
            cls=int(box.cls[0]); cx_px=(x1+x2)/2; cy_px=(y1+y2)/2
            # ── FILTRO: ignorar detecciones fuera de la malla ─────────────
            if not self.cuadricula.punto_dentro(cx_px, cy_px):
                continue
            # ─────────────────────────────────────────────────────────────
            detections.append({
                "bbox":(x1,y1,x2,y2),"cx_px":cx_px,"cy_px":cy_px,
                "cx_norm":cx_px/w,"cy_norm":cy_px/h,"conf":float(box.conf[0]),
                "color":COLORES[cls%len(COLORES)],
                "celda_info":self.cuadricula.info_celda(cx_px,cy_px),
                "esquina_info":self.cuadricula.esquina_mas_cercana(cx_px,cy_px),
            })
        detections.sort(key=lambda x:x["conf"],reverse=True)

        annotated=frame.copy()
        # ── Oscurecer área fuera de la malla ─────────────────────────────
        annotated = self.cuadricula.dibujar_zona_activa(annotated)
        # ─────────────────────────────────────────────────────────────────
        hl_c = detections[0]["celda_info"]["celda"] if detections else None
        hl_e = (detections[0]["esquina_info"]["esquina"]
                if detections and detections[0]["esquina_info"] else None)
        annotated=self.cuadricula.dibujar(annotated,
            highlight_celda=hl_c,celdas_con_demo=celdas_con_demo,
            esquinas_con_demo=esquinas_con_demo,esquina_highlight=hl_e)

        for det in detections:
            x1,y1,x2,y2=[int(v) for v in det["bbox"]]
            ci=det["celda_info"]; ei=det["esquina_info"]; color=det["color"]
            c_s=f"C{ci['celda']}" if ci["celda"] else "fuera"
            e_s=f"E{ei['esquina']}" if ei else "?"
            tag=f"{c_s}/{e_s} {det['conf']:.2f}"
            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
            cx_i,cy_i=int(det["cx_px"]),int(det["cy_px"])
            cv2.circle(annotated,(cx_i,cy_i),6,(0,0,255),-1)
            cv2.circle(annotated,(cx_i,cy_i),6,(255,255,255),1)
            (tw,th),_=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.50,1)
            cv2.rectangle(annotated,(x1,y1-th-8),(x1+tw+4,y1),color,-1)
            cv2.putText(annotated,tag,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,255,255),1)

        if detections:
            x1,y1,x2,y2=[int(v) for v in detections[0]["bbox"]]
            cv2.rectangle(annotated,(x1-3,y1-3),(x2+3,y2+3),(0,255,255),3)
            ci=detections[0]["celda_info"]; ei=detections[0]["esquina_info"]
            txt=(f"OBJETIVO Celda {ci['celda']} | Esquina {ei['esquina']}"
                 if ci["celda"] and ei else "OBJETIVO (fuera de malla)")
            cv2.putText(annotated,txt,(x1,y2+26),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)

        cv2.rectangle(annotated,(0,h-56),(w,h),(0,0,0),-1)
        cv2.putText(annotated,
            f"Trozos:{len(detections)} (solo dentro malla) | ESPACIO=confirmar | ESC/q=salir",
            (8,h-32),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,255,255),1)
        cv2.putText(annotated,
            "Naranja=objetivo | Verde=celda demo | Rojo=esquina demo | Cyan=esquina cercana | Zona oscura=ignorada",
            (8,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
        return detections,annotated


# ─── CAPTURA ──────────────────────────────────────────────────────────────────
def capturar_objetivo(vision,celdas_con_demo,esquinas_con_demo):
    VENTANA="BC TEST  |  ESPACIO=confirmar  |  ESC/q=salir"
    cv2.namedWindow(VENTANA,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA,1100,680); cv2.moveWindow(VENTANA,50,30)
    try: cv2.setWindowProperty(VENTANA,cv2.WND_PROP_TOPMOST,1)
    except: pass

    espera=np.zeros((680,1100,3),dtype=np.uint8)
    cv2.putText(espera,"Iniciando camara...",(290,290),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),2)
    cv2.putText(espera,"Naranja=objetivo | Verde=celda demo | Rojo=esquina demo | Cyan=esquina cercana",
                (60,360),cv2.FONT_HERSHEY_SIMPLEX,0.52,(200,200,200),1)
    cv2.putText(espera,"ESPACIO=confirmar   ESC/q=salir",(310,420),cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),1)
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
        dets,annotated=vision.detectar(frame,celdas_con_demo,esquinas_con_demo)
        cv2.imshow(VENTANA,annotated)
        key=cv2.waitKey(30)&0xFF
        if key in (27,ord('q')): resultado="SALIR"; break
        if key==32:
            if not dets: print("  Sin deteccion YOLO."); continue
            best=dets[0]; ci=best["celda_info"]; ei=best["esquina_info"]
            if ci["celda"] is None: print("  Fuera de malla."); continue
            resultado={"celda_info":ci,"esquina_info":ei,
                       "cx_norm":best["cx_norm"],"cy_norm":best["cy_norm"],"conf":best["conf"]}
            esq_s=f"Esquina {ei['esquina']}" if ei else "sin esquina"
            print(f"  Confirmado: Celda {ci['celda']} | {esq_s} | conf={best['conf']:.2f}")
            break
    cv2.destroyWindow(VENTANA); cv2.waitKey(1)
    return resultado


# ─── EJECUTAR DEMO ────────────────────────────────────────────────────────────
def ejecutar_demo(robot,demo,info_bc):
    pasos=[s for s in demo.get("steps",[]) if s.get("cmd")!="GRIP 0"]
    print(f"\n{'─'*56}")
    print(f"  BC  Celda detectada={info_bc['celda_target']}  "
          f"Celda demo={info_bc['celda_usada']}")
    if info_bc.get('esquina_target'):
        print(f"      Esquina detectada={info_bc['esquina_target']}  "
              f"Esquina demo={info_bc.get('esquina_usada')}")
    print(f"  Fuente: {info_bc['fuente']}  "
          f"{'dist='+str(round(info_bc['distancia'],2)) if info_bc['distancia']>0 else 'exacta'}")
    print(f"  Pasos: {len(pasos)}")
    for s in pasos:
        print(f"    {s['cmd']:<6} {s.get('steps_executed',0):>6}")
    print(f"{'─'*56}\n")

    for i,step in enumerate(pasos,1):
        print(f"  [{i}/{len(pasos)}] ",end="")
        resp=robot.ejecutar_paso(step)
        if resp not in ("OK","SKIP"): print(f"  WARN: {resp}")
        time.sleep(0.05)
    print("  Movimientos OK.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main(simulate=False):
    cuadricula = Cuadricula(CALIB_JSON)
    dataset    = BCDataset(CLEAN_PKL)
    c_demo     = dataset.celdas_con_demo()
    e_demo     = dataset.esquinas_con_demo()
    ciclo      = 0

    with RobotInterface(simulate=simulate) as robot:
        vision = VisionPipeline(cuadricula=cuadricula, simulate=simulate)
        with vision:
            print("="*62)
            print("  BEHAVIOR CLONING TEST  —  malla + esquinas")
            print(f"  {len(dataset.episodes)} demos | {len(c_demo)} celdas | {len(e_demo)} esquinas")
            print("="*62)
            print("\n  Leyenda:")
            print("    Naranja          = celda donde esta el trozo ahora")
            print("    Verde (borde)    = celda con demos grabadas")
            print("    Circulo ROJO     = esquina con demos (numero=cantidad)")
            print("    Circulo CYAN     = esquina mas cercana al trozo")
            print("\n  ESPACIO=confirmar | ESC/q=salir\n")

            continuar = True
            while continuar:
                ciclo += 1
                print(f"\n{'='*62}  CICLO {ciclo}")

                print("\n[1] HOME..."); robot.go_home()
                print("\n[2] Cerrando gripper..."); robot.set_gripper(PINZA_MIN,"captura"); time.sleep(0.4)

                print("\n[3] Abre camara — apunta al plato y presiona ESPACIO...")
                captura = capturar_objetivo(vision,c_demo,e_demo)
                if captura is None or captura=="SALIR":
                    print("  Saliendo..."); continuar=False; break

                ci=captura["celda_info"]; ei=captura["esquina_info"]
                celda_t  = ci["celda"]
                esquina_t = ei["esquina"] if ei else None

                print(f"\n[4] BC lookup: celda={celda_t} esquina={esquina_t}...")
                demo,fuente,celda_u,esq_u,dist = dataset.buscar_demo(celda_t,esquina_t)
                print(f"  -> {fuente}  dist={dist:.2f}")

                info_bc = {"celda_target":celda_t,"celda_usada":celda_u,
                           "esquina_target":esquina_t,"esquina_usada":esq_u,
                           "fuente":fuente,"distancia":dist}

                print("\n[5] Abriendo gripper..."); robot.set_gripper(PINZA_MAX,"pre-agarre"); time.sleep(0.5)
                print("\n[6] Ejecutando demo BC..."); ejecutar_demo(robot,demo,info_bc)
                print("\n[7] Cerrando gripper — AGARRE..."); robot.set_gripper(PINZA_MIN,"agarrando"); time.sleep(0.8)
                print("\n[8] HOME..."); robot.go_home(); time.sleep(0.3)
                print("\n[9] Abriendo — SOLTANDO..."); robot.set_gripper(PINZA_MAX,"soltando"); time.sleep(0.8)
                print("\n[10] Cerrando para siguiente..."); robot.set_gripper(PINZA_MIN,"listo"); time.sleep(0.4)

                print(f"\n  Ciclo {ciclo} OK.")
                print("  ENTER=siguiente | q=salir: ",end="",flush=True)
                try:
                    if input().strip().lower()=="q": continuar=False
                except (EOFError,KeyboardInterrupt):
                    continuar=False

    cv2.destroyAllWindows()
    print("\nPrograma terminado.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--sim",action="store_true")
    parser.add_argument("--calib",default=CALIB_JSON)
    parser.add_argument("--dataset",default=CLEAN_PKL)
    args=parser.parse_args()
    if args.calib!=CALIB_JSON:   globals()["CALIB_JSON"]=args.calib
    if args.dataset!=CLEAN_PKL:  globals()["CLEAN_PKL"]=args.dataset
    main(simulate=args.sim)