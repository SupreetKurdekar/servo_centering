
#include <Servo.h> 

Servo myservo;

void setup() 
{ 
  myservo.attach(9);

} 

void loop() {
  myservo.writeMicroseconds(2000);
  delay(1*1000);
  myservo.writeMicroseconds(2500);
  delay(1*1000);
} 