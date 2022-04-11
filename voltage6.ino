/* EdenOff 
 * power cuts prediction vía Machine Learning
 * TinyML via Edge Impulse
 * Roni Bandini
 * Argentina, April 2022
 * @RoniBandini
 */


#include <Arduino_HTS221.h>
#include <VoltageVariation_inferencing.h>
#include <TM1637Display.h>

// 4 digit display
# define CLK 12 
# define DIO 11
# define Start_Byte 0x7E
# define Version_Byte 0xFF
# define Command_Length 0x06
# define End_Byte 0xEF
# define Acknowledge 0x00 
# define ACTIVATED LOW
# define pinBuzzer 10

TM1637Display display(CLK, DIO);

// Temp
float old_temp = 0;
float old_hum = 0;

// AC
double  sensorValue1 = 0;
double  sensorValue2 = 0;
int     crosscount = 0;
int     climb_flag = 0;
int     val[100];
int     max_v = 0;
double  VmaxD = 0;
double  VeffD = 0;
double  Veff = 0;

// Axis array
float features[3];


int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}


float voltage=0;
float temperature=0;
float avglatest5=0;
float sumVoltage=0;
int   myCounter=1;
int inferenceCounter=0;
float threesold=0.85;
int testFail=0;
int iterationsForAvg=5;

// Degree Celsius symbol
const uint8_t celsius[] = {
  SEG_A | SEG_B | SEG_F | SEG_G,  // Circle
  SEG_A | SEG_D | SEG_E | SEG_F   // C
};

// All segments on
const uint8_t data[] = {0xff, 0xff, 0xff, 0xff};

// All segments off
const uint8_t blank[] = {0x00, 0x00, 0x00, 0x00};

const uint8_t fail[] = {
  SEG_F | SEG_E | SEG_A | SEG_G,           // F
  SEG_E | SEG_F | SEG_A | SEG_B | SEG_C | SEG_G,   // A
  SEG_F | SEG_E ,                           // I
  SEG_F | SEG_E | SEG_D             // L
};

// buzzer beep
void myBeep(){     
     tone(pinBuzzer, 349, 500);
     delay(150);                        
     tone(pinBuzzer, 200, 500);
     delay(150); 
     tone(pinBuzzer, 150, 500);
     delay(500);   
     noTone(pinBuzzer);
}

void setup()
{
    // start display
    display.setBrightness(0x0f); 
    display.setSegments(data);

    pinMode(pinBuzzer, OUTPUT);
    
    // try beep
    myBeep();
    delay(4000);
    
    Serial.begin(115200);
    Serial.println("EdenOff Power Cut Prediction");
    Serial.println("TinyML via Edge Impulse");
    Serial.println("Roni Bandini, April 2022");
    Serial.println("@RoniBandini");
    Serial.println("------------------------------------");

    if (testFail==1){
      Serial.println("Test fail mode enabled");
      }

    if (!HTS.begin()) {
      Serial.println("Failed to init temp sensor :(");
    while (1);
    }
}


void loop()

{
    ei_printf("Reading: ");
    Serial.println(myCounter);

    // read temperature
    temperature = float(HTS.readTemperature());  
  
    // read AC
    for ( int i = 0; i < 100; i++ ) {
          sensorValue1 = analogRead(A0);
          if (analogRead(A0) > 511) {
            val[i] = sensorValue1;
          }
          else {
            val[i] = 0;
          }
          delay(1);
    }

      max_v = 0;
    
      for ( int i = 0; i < 100; i++ )
      {
        if ( val[i] > max_v )
        {
          max_v = val[i];
        }
        val[i] = 0;
      }
      
      if (max_v != 0) {
        VmaxD = max_v;
        VeffD = VmaxD / sqrt(2);
        Veff = (((VeffD - 420.76) / -90.24) * -210.2) + 210.2;
      }
      else {
        Veff = 0;
      }


      VmaxD = 0;
     
    
    if (myCounter==iterationsForAvg+1){                                   

        inferenceCounter++;
        
        if (sumVoltage>0) {
          avglatest5=(sumVoltage)/iterationsForAvg;
        }

        ei_printf("Inference #: ");
        Serial.println(inferenceCounter);
        
        ei_printf("Latest AC: ");
        Serial.println(voltage);

        ei_printf("AVG 5 AC: ");
        Serial.println(avglatest5);

        ei_printf("Temperature: ");
        Serial.println(temperature);

        features[0] = voltage;
        features[1] = temperature;
        features[2] = avglatest5;

        if (sizeof(features) / sizeof(float) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
            ei_printf("The size of your 'features' array is not correct. Expected %lu items, but had %lu\n",
                EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(float));
            delay(1000);
            return;
        }
    
        ei_impulse_result_t result = { 0 };
    
        // the features are stored into flash, and we don't want to load everything into RAM
        signal_t features_signal;
        features_signal.total_length = sizeof(features) / sizeof(features[0]);
        features_signal.get_data = &raw_feature_get_data;
    
        // invoke the impulse
        EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false /* debug */);
        ei_printf("run_classifier returned: %d\n", res);
    
        if (res != 0) return;
    
        // print predictions
        ei_printf("Predictions ");
        ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
        ei_printf(": \n");
        ei_printf("[");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("%.5f", result.classification[ix].value);
          #if EI_CLASSIFIER_HAS_ANOMALY == 1
                  ei_printf(", ");
          #else
                  if (ix != EI_CLASSIFIER_LABEL_COUNT - 1) {
                      ei_printf(", ");
                  }
          #endif
              }
          #if EI_CLASSIFIER_HAS_ANOMALY == 1
              ei_printf("%.3f", result.anomaly);
          #endif
              ei_printf("]\n");
    
          // human-readable predictions
          for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
              ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);

            if (float(result.classification[ix].value)>threesold and result.classification[ix].label=="failure")
            {
              display.setSegments(fail);
              myBeep();                
            }
            else
            {
              if (testFail==1 and (inferenceCounter==3 or inferenceCounter==5 or inferenceCounter==7)){
                // force fail to test display
                display.setSegments(fail);
                myBeep();
              }
            }
            
          }


          
          #if EI_CLASSIFIER_HAS_ANOMALY == 1
              ei_printf("    anomaly score: %.3f\n", result.anomaly);
          #endif

          // reset calculations
          avglatest5=0;
          sumVoltage=0;
          myCounter=0;
        
     }
     else
     {
        // update display
        Serial.print("Voltage: ");
        Serial.println(Veff);       
        display.showNumberDec(int(Veff));
        delay(1000);
        display.showNumberDec(int(temperature), false, 2, 0);
        display.setSegments(celsius, 2, 2);
        delay(1000);
        sumVoltage=sumVoltage+Veff;
      }
           
    delay(1500);
    myCounter=myCounter+1;
}

void ei_printf(const char *format, ...) {
    static char print_buf[1024] = { 0 };

    va_list args;
    va_start(args, format);
    int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
    va_end(args);

    if (r > 0) {
        Serial.write(print_buf);
    }
}
