
#define STEP_PIN   5
#define DIR_PIN    6

#define ENCODER_A  2
#define ENCODER_B  3

#define SLEEP_PIN  8
#define RESET_PIN  9

#define SPEED_LIMIT 40000        


float TARGET_ANGLE = 180.0;
float Kp = 30.0;
float Ki = 0.015;
float Kd = 0.09;

// Encoder resolution
const float CPR = 2400.0;

// PID variables
float prevError = 0;
float integral = 0;
unsigned long lastTime = 0;

// Encoder counter
volatile long encoderCounts = 0;
volatile int lastEncoded = 0;

// High-speed step pulse timing
volatile unsigned long stepInterval = 2000;    // microseconds per step
volatile unsigned long lastStepMicros = 0;


void updateEncoder() {
  int MSB = (PINE & (1 << 4)) >> 4; 
  int LSB = (PINE & (1 << 5)) >> 5; 
  int encoded = (MSB << 1) | LSB;
  int sum = (lastEncoded << 2) | encoded;

  if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderCounts++;
  if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encoderCounts--;

  lastEncoded = encoded;
}


void setMotorSpeed(float speed) {

  if (speed >= 0) digitalWrite(DIR_PIN, HIGH);
  else digitalWrite(DIR_PIN, LOW);

  speed = abs(speed);

  if (speed < 1) speed = 1;
  if (speed > SPEED_LIMIT) speed = SPEED_LIMIT;

  // Convert to time between pulses (microseconds)
  stepInterval = 1000000.0 / speed;
}



void setup() {
  Serial.begin(115200);

  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  pinMode(SLEEP_PIN, OUTPUT);
  pinMode(RESET_PIN, OUTPUT);
  digitalWrite(SLEEP_PIN, HIGH);
  digitalWrite(RESET_PIN, HIGH);

  pinMode(ENCODER_A, INPUT_PULLUP);
  pinMode(ENCODER_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENCODER_A), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER_B), updateEncoder, CHANGE);

  lastTime = micros();
}


void loop() {


  unsigned long now = micros();
  float dt = (now - lastTime) / 1000000.0;
  lastTime = now;


  float angle = abs(fmod((encoderCounts / CPR) * 360.0, 360.0));


  float error = angle - TARGET_ANGLE;
  error = -error;

  integral += error * dt;
  integral = constrain(integral, -5000, 5000);

  float derivative = (error - prevError) / dt;
  prevError = error;

  float pid = Kp * error + Ki * integral + Kd * derivative;
  
  Serial.println(angle);
  // Don't drive outside upright region
  if (angle < 135 || angle > 225) return;

  // Motor speed update
  setMotorSpeed(pid);



  unsigned long microsNow = micros();
  if (microsNow - lastStepMicros >= stepInterval) {
    lastStepMicros = microsNow;

    // Fast pulse (2–3 µs)
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(2);
    digitalWrite(STEP_PIN, LOW);
  }
}