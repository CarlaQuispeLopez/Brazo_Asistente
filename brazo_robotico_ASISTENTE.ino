// ============================================================
//   BRAZO ROBOTICO - RAMPS 1.4 + Arduino Mega 2560
//   Libreria necesaria: AccelStepper (instalar en Library Manager)
//
//   COMANDOS (Serial Monitor a 115200 baudios):
//     BASE 200        mueve la base
//     HOMBRO 100      sube/baja el hombro
//     CODO 100        dobla el codo
//     GRIPPER 100     sube/baja el gripper
//     GIRO 100        gira el gripper
//     PINZA ABRIR     abre la pinza
//     PINZA CERRAR    cierra la pinza
//     PINZA 45        angulo exacto (0 a 90)
//     PARAR           detiene todo
//     POSICION        ver posicion actual
// ============================================================

#include <AccelStepper.h>
#include <Servo.h>

// --- PINES (RAMPS 1.4) ---
// BASE → Motor Z
#define BASE_STEP    46
#define BASE_DIR     48
#define BASE_EN      62

// HOMBRO → Motor Y
#define HOMBRO_STEP  60
#define HOMBRO_DIR   61
#define HOMBRO_EN    56

// CODO → Motor X
#define CODO_STEP    54
#define CODO_DIR     55
#define CODO_EN      38

// GRIPPER SUBE/BAJA → Motor E0
#define GRIPPER_STEP 26
#define GRIPPER_DIR  28
#define GRIPPER_EN   24

// GRIPPER GIRO → Motor E1
#define GIRO_STEP    36
#define GIRO_DIR     34
#define GIRO_EN      30

// SERVO PINZA
#define SERVO_PIN    11

// --- VELOCIDADES (pasos/seg) y ACELERACION (pasos/seg²) ---
#define VEL_BASE     300.0
#define VEL_HOMBRO   250.0
#define VEL_CODO     250.0
#define VEL_GRIPPER  350.0
#define VEL_GIRO     350.0
#define ACEL_NORMAL  120.0
#define ACEL_PESADO   70.0

// --- SERVO: solo de 0 a 90 grados ---
#define PINZA_CERRADA  0
#define PINZA_ABIERTA 90

// --- INSTANCIAS ---
AccelStepper mBase    (AccelStepper::DRIVER, BASE_STEP,    BASE_DIR);
AccelStepper mHombro  (AccelStepper::DRIVER, HOMBRO_STEP,  HOMBRO_DIR);
AccelStepper mCodo    (AccelStepper::DRIVER, CODO_STEP,    CODO_DIR);
AccelStepper mGripper (AccelStepper::DRIVER, GRIPPER_STEP, GRIPPER_DIR);
AccelStepper mGiro    (AccelStepper::DRIVER, GIRO_STEP,    GIRO_DIR);

Servo servoPinza;
int anguloPinza = 0;

// ============================================================
void setup() {
  Serial.begin(115200);

  // Habilitar drivers (LOW = activo en A4988)
  pinMode(BASE_EN,    OUTPUT); digitalWrite(BASE_EN,    LOW);
  pinMode(HOMBRO_EN,  OUTPUT); digitalWrite(HOMBRO_EN,  LOW);
  pinMode(CODO_EN,    OUTPUT); digitalWrite(CODO_EN,    LOW);
  pinMode(GRIPPER_EN, OUTPUT); digitalWrite(GRIPPER_EN, LOW);
  pinMode(GIRO_EN,    OUTPUT); digitalWrite(GIRO_EN,    LOW);

  // Configurar velocidad y aceleracion suave
  mBase.setMaxSpeed(VEL_BASE);       mBase.setAcceleration(ACEL_PESADO);
  mHombro.setMaxSpeed(VEL_HOMBRO);   mHombro.setAcceleration(ACEL_PESADO);
  mCodo.setMaxSpeed(VEL_CODO);       mCodo.setAcceleration(ACEL_PESADO);
  mGripper.setMaxSpeed(VEL_GRIPPER); mGripper.setAcceleration(ACEL_NORMAL);
  mGiro.setMaxSpeed(VEL_GIRO);       mGiro.setAcceleration(ACEL_NORMAL);

  // Servo: pinza cerrada al encender
  servoPinza.attach(SERVO_PIN);
  delay(100);
  servoPinza.write(PINZA_CERRADA);
  anguloPinza = PINZA_CERRADA;

  Serial.println(F("BRAZO LISTO - todos en reposo - pinza cerrada"));
}

// ============================================================
void loop() {
  mBase.run();
  mHombro.run();
  mCodo.run();
  mGripper.run();
  mGiro.run();

  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() > 0) procesarComando(cmd);
  }
}

// ============================================================
void procesarComando(String cmd) {
  cmd.toUpperCase();
  int sp       = cmd.indexOf(' ');
  String acc   = (sp == -1) ? cmd         : cmd.substring(0, sp);
  String val   = (sp == -1) ? String("") : cmd.substring(sp + 1);
  val.trim();

  if      (acc == "BASE")     moverMotor(mBase,    "BASE",    val.toInt(), VEL_BASE);
  else if (acc == "HOMBRO")   moverMotor(mHombro,  "HOMBRO",  val.toInt(), VEL_HOMBRO);
  else if (acc == "CODO")     moverMotor(mCodo,    "CODO",    val.toInt(), VEL_CODO);
  else if (acc == "GRIPPER")  moverMotor(mGripper, "GRIPPER", val.toInt(), VEL_GRIPPER);
  else if (acc == "GIRO")     moverMotor(mGiro,    "GIRO",    val.toInt(), VEL_GIRO);
  else if (acc == "PINZA") {
    if      (val == "ABRIR")  moverPinza(PINZA_ABIERTA);
    else if (val == "CERRAR") moverPinza(PINZA_CERRADA);
    else                      moverPinza(constrain(val.toInt(), PINZA_CERRADA, PINZA_ABIERTA));
  }
  else if (acc == "PARAR")    pararTodo();
  else if (acc == "POSICION") mostrarPosicion();
  else Serial.println(F("Comando no reconocido"));
}

// ============================================================
void moverMotor(AccelStepper &motor, const char* nombre, int pasos, float vel) {
  motor.setMaxSpeed(vel);
  motor.move(pasos);
  Serial.print(nombre); Serial.print(F(": ")); Serial.print(pasos); Serial.println(F(" pasos..."));
  while (motor.distanceToGo() != 0) {
    motor.run();
    if (Serial.available()) {
      String s = Serial.readStringUntil('\n');
      s.trim(); s.toUpperCase();
      if (s == "PARAR") { motor.stop(); while (motor.distanceToGo() != 0) motor.run(); Serial.println(F("Detenido.")); return; }
    }
  }
  Serial.println(F("OK"));
}

void moverPinza(int dest) {
  dest = constrain(dest, PINZA_CERRADA, PINZA_ABIERTA);
  int paso = (dest > anguloPinza) ? 1 : -1;
  Serial.print(F("Pinza: ")); Serial.print(anguloPinza); Serial.print(F(" -> ")); Serial.print(dest); Serial.println(F(" grados"));
  while (anguloPinza != dest) { anguloPinza += paso; servoPinza.write(anguloPinza); delay(12); }
  Serial.println(F("OK"));
}

void pararTodo() {
  mBase.stop(); mHombro.stop(); mCodo.stop(); mGripper.stop(); mGiro.stop();
  Serial.println(F("TODOS DETENIDOS"));
}

void mostrarPosicion() {
  Serial.println(F("--- POSICION ---"));
  Serial.print(F("BASE:    ")); Serial.println(mBase.currentPosition());
  Serial.print(F("HOMBRO:  ")); Serial.println(mHombro.currentPosition());
  Serial.print(F("CODO:    ")); Serial.println(mCodo.currentPosition());
  Serial.print(F("GRIPPER: ")); Serial.println(mGripper.currentPosition());
  Serial.print(F("GIRO:    ")); Serial.println(mGiro.currentPosition());
  Serial.print(F("PINZA:   ")); Serial.print(anguloPinza); Serial.println(F(" grados"));
  Serial.println(F("----------------"));
}
