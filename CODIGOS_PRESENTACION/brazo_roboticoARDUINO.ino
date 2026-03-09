#include <AccelStepper.h>
#include <Servo.h>

#define BASE_STEP    46
#define BASE_DIR     48
#define BASE_EN      62

#define HOMBRO_STEP  60
#define HOMBRO_DIR   61
#define HOMBRO_EN    56

#define CODO_STEP    54
#define CODO_DIR     55
#define CODO_EN      38

#define GRIPPER_STEP 26
#define GRIPPER_DIR  28
#define GRIPPER_EN   24

#define GIRO_STEP    36
#define GIRO_DIR     34
#define GIRO_EN      30

#define SERVO_PIN    11
#define PINZA_CERRADA  0
#define PINZA_ABIERTA 90

#define VEL_BASE     300.0
#define VEL_HOMBRO   250.0
#define VEL_CODO     250.0
#define VEL_GRIPPER  350.0
#define VEL_GIRO     350.0
#define ACEL_NORMAL  120.0
#define ACEL_PESADO   70.0

AccelStepper mBase    (AccelStepper::DRIVER, BASE_STEP,    BASE_DIR);
AccelStepper mHombro  (AccelStepper::DRIVER, HOMBRO_STEP,  HOMBRO_DIR);
AccelStepper mCodo    (AccelStepper::DRIVER, CODO_STEP,    CODO_DIR);
AccelStepper mGripper (AccelStepper::DRIVER, GRIPPER_STEP, GRIPPER_DIR);
AccelStepper mGiro    (AccelStepper::DRIVER, GIRO_STEP,    GIRO_DIR);

Servo servoPinza;
int anguloPinza = 0;

void setup() {
  Serial.begin(115200);

  pinMode(BASE_EN,    OUTPUT); digitalWrite(BASE_EN,    LOW);
  pinMode(HOMBRO_EN,  OUTPUT); digitalWrite(HOMBRO_EN,  LOW);
  pinMode(CODO_EN,    OUTPUT); digitalWrite(CODO_EN,    LOW);
  pinMode(GRIPPER_EN, OUTPUT); digitalWrite(GRIPPER_EN, LOW);
  pinMode(GIRO_EN,    OUTPUT); digitalWrite(GIRO_EN,    LOW);

  mBase.setMaxSpeed(VEL_BASE);       mBase.setAcceleration(ACEL_PESADO);
  mHombro.setMaxSpeed(VEL_HOMBRO);   mHombro.setAcceleration(ACEL_PESADO);
  mCodo.setMaxSpeed(VEL_CODO);       mCodo.setAcceleration(ACEL_PESADO);
  mGripper.setMaxSpeed(VEL_GRIPPER); mGripper.setAcceleration(ACEL_NORMAL);
  mGiro.setMaxSpeed(VEL_GIRO);       mGiro.setAcceleration(ACEL_NORMAL);

  servoPinza.attach(SERVO_PIN);
  delay(100);
  servoPinza.write(PINZA_CERRADA);
  anguloPinza = PINZA_CERRADA;

  Serial.println(F("BRAZO LISTO"));
}

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
  while (anguloPinza != dest) { anguloPinza += paso; servoPinza.write(anguloPinza); delay(12); }
  Serial.println(F("OK"));
}

void pararTodo() {
  mBase.stop(); mHombro.stop(); mCodo.stop(); mGripper.stop(); mGiro.stop();
  Serial.println(F("DETENIDO"));
}

void mostrarPosicion() {
  Serial.print(F("BASE:")); Serial.println(mBase.currentPosition());
  Serial.print(F("HOMBRO:")); Serial.println(mHombro.currentPosition());
  Serial.print(F("CODO:")); Serial.println(mCodo.currentPosition());
  Serial.print(F("GRIPPER:")); Serial.println(mGripper.currentPosition());
  Serial.print(F("GIRO:")); Serial.println(mGiro.currentPosition());
  Serial.print(F("PINZA:")); Serial.println(anguloPinza);
}

void procesarComando(String cmd) {
  cmd.toUpperCase();
  int sp     = cmd.indexOf(' ');
  String acc = (sp == -1) ? cmd : cmd.substring(0, sp);
  String val = (sp == -1) ? String("") : cmd.substring(sp + 1);
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
