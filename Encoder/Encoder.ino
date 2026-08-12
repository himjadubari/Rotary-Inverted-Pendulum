// --- CONSTANTS ---
// CHECK THIS: How many 'counts' for one full 360 turn?
// Common values: 1200, 2400, 4000
const float CPR = 3000.0; 

// --- PIN DEFINITIONS ---
#define ENCODER_A 18
#define ENCODER_B 19

volatile long encoderValue = 0;
volatile int lastEncoded = 0;

void setup() {
  Serial.begin(9600);
  
  pinMode(ENCODER_A, INPUT_PULLUP);
  pinMode(ENCODER_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENCODER_A), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER_B), updateEncoder, CHANGE);
}

void loop() {
  // 1. Calculate Angle
  // We divide current count by Total Counts per Rev, then multiply by 360
  float angle = (encoderValue / CPR) * 360.0;

  // 2. Normalize to 0-360 (Remove full rotations)
  // fmod is the "modulo" operator for decimal numbers
  angle = fmod(angle, 360.0);

  // 3. Fix Negatives (Make -90 into 270)
  if (angle < 0) {
    angle += 360.0;
  }

  // 4. Print
  Serial.print("Raw Count: ");
  Serial.print(encoderValue);
  Serial.print(" | Angle: ");
  Serial.println(angle); // This will print 0.00 to 359.99

  delay(50); // Small delay to make it readable
}

// --- INTERRUPT ROUTINE ---
void updateEncoder() {
  int MSB = digitalRead(ENCODER_A); 
  int LSB = digitalRead(ENCODER_B); 

  int encoded = (MSB << 1) | LSB; 
  int sum  = (lastEncoded << 2) | encoded; 

  if(sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderValue++;
  if(sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encoderValue--;

  lastEncoded = encoded; 
}