// =====================================================
//           LQR CONTROL – INVERTED PENDULUM
//        For Arduino UNO + Stepper Motor (Nema 17)
// =====================================================

#define STEP_PIN   5
#define DIR_PIN    6
#define ENCODER_A  2
#define ENCODER_B  3
#define SLEEP_PIN  8
#define RESET_PIN  9

#define SPEED_LIMIT 40000.0 // Max Step Speed

// --- LQR GAINS ---
// You usually calculate these in MATLAB/Python based on physics.
// These are manual estimates that act like a superior PID.
// State vector: [Pendulum_Angle, Pendulum_Vel, Arm_Position, Arm_Vel]

// k1 (Angle Penalty): Keeps it upright. (Like PID 'P')
float k1 = 200.0; 

// k2 (Ang Vel Penalty): Damping/Stability. (Like PID 'D')
float k2 = 50.0;  

// k3 (Arm Pos Penalty): KEEPS THE ARM CENTERED. (PID doesn't do this!)
float k3 = 2.5;   

// k4 (Arm Vel Penalty): Prevents arm from whipping too fast.
float k4 = 1.5;   

// --- SYSTEM CONSTANTS ---
const float CPR = 2400.0;     // Encoder Counts per Rev
const float STEPS_PER_REV = 400.0; // Stepper steps (assuming 1/2 microstep)
// NOTE: Ensure your stepper driver is set to the microstepping matching this.

// --- VARIABLES ---
// Pendulum State
volatile long encCounts = 0;
volatile int lastEncoded = 0;
float pendAngle = 0;
float pendVel = 0;
float lastPendAngle = 0;

// Motor State
long stepperPosition = 0; // Tracks arm angle relative to start
float motorVel = 0;
volatile long stepInterval = 1000000;
volatile bool motorEnabled = false;
volatile bool dirForward = true;

// Timing
unsigned long lastControlTime = 0;
unsigned long lastStepMicros = 0;
const int CONTROL_LOOP_MS = 5; 

// Filter
float filteredPendVel = 0;
const float ALPHA = 0.4;

// Swing Up
float swingRamp = 0.0;
float TARGET_ANGLE = 180.0; // Upright

// -----------------------------------------------------
//                  ENCODER ISR
// -----------------------------------------------------
void updateEncoder() {
  int MSB = (PIND & (1 << 2)) >> 2; 
  int LSB = (PIND & (1 << 3)) >> 3; 
  int encoded = (MSB << 1) | LSB;
  int sum = (lastEncoded << 2) | encoded;

  if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encCounts++;
  if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encCounts--;
  lastEncoded = encoded;
}

// -----------------------------------------------------
//              STEPPER CONTROL
// -----------------------------------------------------
void setSpeed(float speed) {
  if (abs(speed) < 10) {
    motorEnabled = false;
    return;
  }
  motorEnabled = true;
  
  // Update direction and track position
  if (speed > 0) {
    digitalWrite(DIR_PIN, HIGH);
    dirForward = true;
  } else {
    digitalWrite(DIR_PIN, LOW);
    dirForward = false;
  }

  float absSpeed = abs(speed);
  if (absSpeed > SPEED_LIMIT) absSpeed = SPEED_LIMIT;
  stepInterval = (unsigned long)(1000000.0 / absSpeed);
  
  // Store velocity for LQR feedback
  motorVel = speed; 
}

void enableDriver(bool state) {
  if (state) {
    digitalWrite(SLEEP_PIN, HIGH);
    digitalWrite(RESET_PIN, HIGH);
    delayMicroseconds(10);
  } else {
    digitalWrite(SLEEP_PIN, LOW);
    digitalWrite(RESET_PIN, LOW);
    motorEnabled = false;
  }
}

// -----------------------------------------------------
//                      SETUP
// -----------------------------------------------------
void setup() {
  Serial.begin(115200);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(SLEEP_PIN, OUTPUT);
  pinMode(RESET_PIN, OUTPUT);
  
  enableDriver(false); // Start Off

  pinMode(ENCODER_A, INPUT_PULLUP);
  pinMode(ENCODER_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_A), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER_B), updateEncoder, CHANGE);

  Serial.println("LQR Controller Ready.");
  Serial.println("Calibrating... Hold DOWN.");
  delay(2000);
  noInterrupts();
  encCounts = 0; // Zero is Down
  interrupts();
  Serial.println("Done. 0=Down, 180=Up");
}

// -----------------------------------------------------
//                    MAIN LOOP
// -----------------------------------------------------
void loop() {
  
  // 1. FAST STEPPER PULSE GENERATOR
  if (motorEnabled) {
    unsigned long now = micros();
    if (now - lastStepMicros >= stepInterval) {
      lastStepMicros = now;
      
      // Pulse
      digitalWrite(STEP_PIN, HIGH);
      delayMicroseconds(2);
      digitalWrite(STEP_PIN, LOW);
      
      // Track Arm Position (Essential for LQR)
      if (dirForward) stepperPosition++;
      else stepperPosition--;
    }
  }

  // 2. CONTROL LOOP (5ms / 200Hz)
  unsigned long nowMs = millis();
  if (nowMs - lastControlTime >= CONTROL_LOOP_MS) {
    float dt = (nowMs - lastControlTime) / 1000.0;
    lastControlTime = nowMs;

    // --- STATE ESTIMATION ---
    long counts;
    noInterrupts(); counts = encCounts; interrupts();
    
    // State 1: Pendulum Angle (Normalized -180 to 180)
    pendAngle = (counts / CPR) * 360.0;
    while (pendAngle > 180) pendAngle -= 360;
    while (pendAngle <= -180) pendAngle += 360;
    
    // State 2: Pendulum Velocity (Filtered)
    float rawVel = (pendAngle - lastPendAngle) / dt;
    // Handle wrap-around noise
    if (abs(pendAngle - lastPendAngle) > 180) rawVel = 0; 
    
    filteredPendVel = (ALPHA * rawVel) + ((1.0 - ALPHA) * filteredPendVel);
    lastPendAngle = pendAngle;

    // State 3: Arm Angle (Normalized to degrees, 0 is start center)
    float armAngle = (stepperPosition / STEPS_PER_REV) * 360.0;
    
    // State 4: Arm Velocity (Taken from previous command or calculated)
    // We use the variable 'motorVel' which we commanded previously.

    // --- ERROR CALCULATION ---
    // Target is 180 (Up). 
    // Error is defined as (Current - Target). 
    // If Angle is 170, Error is -10. LQR wants to drive Error to 0.
    float angleError = 0;
    if (pendAngle > 0) angleError = 180.0 - pendAngle;
    else angleError = -180.0 - pendAngle;
    
    float controlOutput = 0;

    // --- DECISION: LQR or SWING UP? ---
    if (abs(angleError) < 40.0) {
       // *** LQR CONTROL LAW ***
       // u = -Kx
       // We want to force:
       // angleError -> 0
       // pendVel -> 0
       // armAngle -> 0 (Center of table)
       // motorVel -> 0
       
       swingRamp = 0.0;
       
       // Calculate Control Effort (u)
       // Note: Signs depend on motor direction vs sensor direction.
       // Ideally: u = (k1 * error) - (k2 * vel) - (k3 * armPos) - (k4 * armVel)
       
       float term1 = k1 * angleError;       // Correct angle
       float term2 = k2 * -filteredPendVel; // Damping
       float term3 = k3 * -armAngle;        // Center the arm (Position Control)
       float term4 = k4 * -motorVel;        // Dampen motor accel
       
       controlOutput = term1 + term2 + term3 + term4;
       
       enableDriver(true);

    } else {
       // *** SWING UP (Standard Energy Pumping) ***
       // Disable arm centering logic during swing up
       
       // Simple Bang-Bang Boost
       swingRamp += 0.005;
       if (swingRamp > 1.0) swingRamp = 1.0;
       
       // If stuck at bottom
       if (abs(filteredPendVel) < 10 && abs(angleError) > 160) {
         controlOutput = 1500 * swingRamp;
       } else {
         float direction = (filteredPendVel > 0) ? 1.0 : -1.0;
         controlOutput = (filteredPendVel * 1400.0) + (direction * 600.0);
         controlOutput *= swingRamp;
       }
       
       enableDriver(true);
    }

    // --- SAFETY ---
    // If robot fell, output is naturally handled by Swing Up, 
    // but if we want a hard kill-switch for testing:
    // if (abs(pendAngle) < 90) enableDriver(false);

    // Apply Speed
    setSpeed(controlOutput);
    
    // Debug
    static unsigned long lastPrint = 0;
    if (millis() - lastPrint > 150) {
      Serial.print("AngErr:"); Serial.print(angleError, 1);
      Serial.print(" ArmPos:"); Serial.print(armAngle, 1);
      Serial.print(" Cmd:"); Serial.println(controlOutput);
      lastPrint = millis();
    }
  }
}