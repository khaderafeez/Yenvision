 ## Introduction
   In today's world, social interactions and emotional understanding are crucial for personal and professional success. However, many individuals with Autism Spectrum Disorder (ASD) face significant challenges in recognizing and interpreting facial expressions, making these interactions more difficult. This project aims to address these challenges by developing an innovative, user-friendly solution tailored to the needs of those with ASD. Leveraging advanced technologies such as augmented reality , we aim to create engaging and personalized tools that provide real-time feedback and immersive learning experiences. By enhancing the ability to recognize emotions, this project seeks to improve social connectivity and overall well-being for individuals with ASD, helping them navigate the complexities of today's social landscape more effectively.

 ## Overview
   In today's world, individuals with Autism Spectrum Disorder (ASD) often struggle with recognizing facial expressions, which impacts their social interactions and emotional development. Existing tools often lack engagement and personalized support. This project addresses these challenges by developing a Smart Interactive Device using Arduino Uno. The device utilizes RFID cards to display emotions, which are read and processed for visual and tactile feedback. Stress relief smiley balls with sensors provide interactive and calming feedback. Integrated into educational and therapeutic settings, the device facilitates learning through games and activities designed to improve facial expression recognition and enhance social interaction skills, aiming to improve emotional intelligence and overall well-being for individuals with ASD.
- **Arduino Uno Microcontroller**: Central processing unit for device operations.
- **RFID Cards**: Display emotions and interact with the system for recognition.
- **Stress Relief Smiley Balls**: Provide tactile feedback and interaction.
- **Buzzer and LED Indicators**: Offer auditory and visual cues for feedback.
- **Push Buttons**: Allow users to select emotions and interact with the device.
- **JQ6500 MP3 Voice Sound Module**: Enhances learning with audio cues.
- **Monitor**: Displays emotions and feedback to the user.

# Components Required

| Item Name                    | Quantity | Description           | Links to Product |
|------------------------------|----------|-----------------------|------------------|
| Arduino Uno Microcontroller  | 1        | Central processor     | [Ardunio uno](https://www.amazon.in/Board-ATmega328P-Cable-Compatible-Projects/dp/B0CY2QDQHW/ref=sr_1_12?dib=eyJ2IjoiMSJ9.gKff2cZc-TTQtoNrhQ-fPaD2yATcW4TtfCFR0GGWbZYLLpZcpfNhKFTUIzMi0Yybnvno5s8mH-65Q0ZhF2sRyvWlGNZSI6uHwBWCoveSDZjFt_bvXMUQrba-LFF88cJyIshxhZWjW0YCScBVlfuTFVFD8AGNVVON2fkH-5D3R6Hfn-WK58CJ1V5WPGM7ZUF0aqJVGpu_lrJ-CKjTgL2t-MNQAOiCF3HbYdbLsxWoYQ4s-NFCIC-0jQh7zhtLs71RWq1bgzw8OgFRa8K-z1eXS1f9yl3nUtD3k3zujv-R2-0.ncKbGl6YvJAXDS7DUr9VCM5WJ4PafkcQVDFSXs8H06Q&dib_tag=se&keywords=arduino+uno&qid=1721052538&sr=8-12)                 |
| RFID Reader                  | 1        | Reads RFID cards      |[RFID Reader](https://www.amazon.in/ApTechDeals-RFID-Kit-Arduino-Raspberry/dp/B07Q1B6QZR/ref=sr_1_1?crid=3ACRMO7I730KG&dib=eyJ2IjoiMSJ9.QFel9f2vOf2vDbvVAqUVl9QpX_uyGB9UovkPYqJ1X7krUmCe7umcx0IDHPlL6FcGT3x0zvAAAHKxGZ5Z4b2dx99drjbtLI89Mb4E11dvvZkZ2KBCcGrqtf6Nu1YYRHGecYMuhtLG4pYbVezMT--PB8OQnicn0KK3RBnOPjgQcF7d2rA3PpJf3q0e1yf6Ap6kkVVjjZdcsJn5V5LZNj8pax5yRN8uEvvoYnNa_SImR76BgZFFpY6Rma9bZ9I5pfF_DN_yGxluYSQuZLoBKEI7qXwQ7dXrKqlXUKKonrjQJ0k.QCcoJgAJ8KyMz8pjxwK036czfVgsX4yQxXkwyDiCVfY&dib_tag=se&keywords=rfid+module+for+arduino&qid=1721052753&sprefix=rfid+modul%2Caps%2C220&sr=8-1)                  |
| Buzzer                       | 1        | Auditory feedback     |[Buzzer](https://www.amazon.in/SOOTRA-Passive-Acoustic-Component-electronic/dp/B0BH9D86KW/ref=sr_1_4?crid=14RFYW546F7BK&dib=eyJ2IjoiMSJ9.GsgZf7vKaslHPdONGuZ_LMksRoG1uT61dkCSHJol0nzag6mfWetoW3vfm8Wj31wNuIyLZ0hGh5At9t8xMr9XlV8OfhkqyWdny4jf8HgRI1ODKabrkLTYSEIO7ndZeeKrPXRufWgJIOavGhYqW-h6NjQjjkrpSiACbILfvmuiyQDcjFtUqH5No5x6fB1zzHHJea5ZuOP2g2NZ3wp-r0vt64N4JiXStZER9c_dCa4PNDxtl2LUSM_-GXPjRwu3t80iLwVWf0lzAHkWGMIFWcMjU4pub7e7KbTY0JRPYtzlIQQ.ib3cuE-sS0B60jKJj83eu_bVTT8QNQM5ppxsG5evzcw&dib_tag=se&keywords=small%2Bbuzzer%2Bfor%2Bproject&qid=1721052835&sprefix=small%2Bbuzz%2Caps%2C216&sr=8-4&th=1)                  |
| LED Indicators               | 1        | Visual feedback       |[LED](https://www.amazon.in/Electronic-Component-Diffused-Indicator-Experiment/dp/B09YYJTKV8/ref=sr_1_8?crid=13MAAAXNVPEGY&dib=eyJ2IjoiMSJ9.4REGGvZvVBsQ72LyxGSdz2CNS5nhj93o-0QjOJX0w2vmP6Ng1d0GvkREAzuIjrs4G0FrxqqbTpY9Z8OL9IE_jODdlJOecgg7228D5FjNV5AAk7v-IHkLVXkSuSAHmYIj6bldrCZB9mPv7QSj_1mhJ6xjK7sNmfcY7pM0d4Ze0Jn3JUU88UOIv49L8XYpqQru1m9IwmoPIFp2cxZdmvDoYqKV5CKw4126DmpA9Q88Mt7UEDe-8128w2BY3NFEP5cbyKEBml95MPSyYs42U7hJdmpw5-BmtgQkH0GV1cGhhVg.CJeClLgTDf1FzvH3TpEUnf5e2zxKuUW6-wM30gxU8tU&dib_tag=se&keywords=led+diode&qid=1721052970&sprefix=led+diod%2Caps%2C216&sr=8-8)                  |
| Push Buttons                 | 4        | User interaction      |[Push Buttons](https://www.amazon.in/SRS-12x12x7-3-Electronics-Peripherals-Industrial/dp/B0BXXFWNTY/ref=sr_1_3?crid=3A2J3L3YY0I61&dib=eyJ2IjoiMSJ9._hTYgaqLdZErVnE0lMtbWq2t1d1YZlkXl9qGW9sKW6sCQKteO687F-QEa8yyMbo4ZlOfx2IqhjeeJcPdTV5JVGHYI27ka9owvfLg9NzobM7gbtSTy7f6KdhDzku2S0DBSW1XY278pzEk9wqrbZ3UVtTZgBWNikfbdPMdh6ti0N-LoWNUXEzHLK89DyGkJS-0pTFGkf24Evz0g8oPXjUErltdv5oazww48L2SnP0K4eexDLGKXWEpGoJ8e3iWcPk4hLyCXwhWW3ZRc4Waa78UaxmGZzSKKC9djcNS9rAbTvo.BvN6uyzUmRQLEAZ4-NhOmGp7YqLUnJkiuWde_Ao1lng&dib_tag=se&keywords=4+push+button&qid=1721053053&sprefix=4+push+button%2Caps%2C212&sr=8-3)                  |
| OLED Display                  | 1        | Displays emotions     |[OLED Display](https://www.amazon.in/Electronic-Spices-Display-Compatible-Arduino/dp/B0CC244ZFM/ref=sr_1_1?crid=1T6JY39OW6067&dib=eyJ2IjoiMSJ9.a-DojWSyNXw_YbJrYbBAnGB2MIllYKKTSGnZxeedQoDOPOsOZx-ZjFCLp81LQp1MCEU5chipOAwTMGR4giPbMgUtt1D__SKBMHspqhAs_XCyHQ7zbl_NKy3KZC0YUkhMtO1hFqeOCW8QtgSUOIk0BfR7PzDR77ZJCxxN4DVuoK6bH6PLMDSE65n5ut8sLwns1AzYXSifUNACnTRrjBNSqnggJRuaOt65yeVUNOOMq8I.Qo0XIi1dZumKv56ZTq5J_JB4PViqvaP9CcYT_pF1PO8&dib_tag=se&keywords=oled+display+for+arduino&qid=1721053101&sprefix=oled+displ%2Caps%2C215&sr=8-1)                  |


# Pin Connections

| Component           | Arduino Uno Pin | Description                    |
|---------------------|-----------------|--------------------------------|
| RFID Reader (SPI)   | 10 (SS), 9 (RST)| Communicates with RFID cards    |
| Buzzer              | 1               | Provides auditory feedback      |
| LED Indicators      | 2 (red), 3 (green) | Provides visual feedback     |
| Push Buttons        | 4, 5, 6, 7, 8   | Interacts with user for emotion selection |

## Working Code
```

#include <SPI.h>
#include <MFRC522.h>

// Define the pins used by the RFID reader
#define RST_PIN 9
#define SS_PIN 10

// Create an instance of the MFRC522 library
MFRC522 rfid(SS_PIN, RST_PIN);

// Define the RFID to string mapping
const int numMappings = 5;
const byte mappings[numMappings][4] = {
  {0x5A, 0xF0, 0x0B, 0xB0},
  {0x1C, 0x0F, 0x33, 0x49},
  {0x53, 0x77, 0x08, 0x50},
  {0x53, 0xF9, 0x4A, 0xAC},
  {0xDE, 0x5C, 0xAE, 0x1D}
};
const char* strings[numMappings] = {"happy", "sad", "fear", "angry", "neutral"};

const int button1Pin = 8;
const int button2Pin = 7;
const int button3Pin = 6;
const int button4Pin = 5;
const int button5Pin = 4;
const int greenLedPin = 3;
const int redLedPin = 2;
const int buzzerPin = 1;

String userInput = "";

void setup() {
  Serial.begin(9600);
  SPI.begin();
  rfid.PCD_Init();

  pinMode(button1Pin, INPUT_PULLUP);
  pinMode(button2Pin, INPUT_PULLUP);
  pinMode(button3Pin, INPUT_PULLUP);
  pinMode(button4Pin, INPUT_PULLUP);
  pinMode(button5Pin, INPUT_PULLUP);
  pinMode(greenLedPin, OUTPUT);
  pinMode(redLedPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);

  digitalWrite(greenLedPin, LOW);
  digitalWrite(redLedPin, LOW);
  digitalWrite(buzzerPin, LOW);
}

void playNote(int noteFrequency, int duration) {
  tone(buzzerPin, noteFrequency, duration);
  delay(duration + 50);
  noTone(buzzerPin);
}

void loop() {
  // Check if a new RFID card is available
  if (rfid.PICC_IsNewCardPresent()) {
    // Verify if the NUID has been readed
    if (rfid.PICC_ReadCardSerial()) {
      // Check if the RFID UID matches any of the mappings
      bool found = false;
      for (int i = 0; i < numMappings; i++) {
        if (memcmp(rfid.uid.uidByte, mappings[i], sizeof(mappings[i])) == 0) {
          userInput = strings[i];
          found = true;
          break;
        }
      }

      // If no matching m apping is found, print a message
      if (!found) {
        Serial.println("Unknown RFID UID");
        userInput = "";
      }

      // Halt the PICC (RFID tag) to make it inactive
      rfid.PICC_HaltA();
    }
  }

  if (!userInput.equals("")) {
    Serial.print("Press a RFID card of emotion '" + userInput + "': ");

    bool greenLedOn = false;
    bool redLedOn = false;

    while (true) {
      if (digitalRead(button1Pin) == LOW) {
        Serial.println("Button pressed: 1");
        if (userInput.equalsIgnoreCase("happy")) {
          Serial.println("Matched: " + userInput);
          greenLedOn = true;
          playNote(400, 500); // Play a higher frequency for "happy"
        } else {
          Serial.println("Not matched");
          redLedOn = true;
          playNote(150, 500);
        }
        delay(1000);
        break;
      } else if (digitalRead(button2Pin) == LOW) {
        Serial.println("Button pressed: 2");
        if (userInput.equalsIgnoreCase("sad")) {
          Serial.println("Matched: " + userInput);
          greenLedOn = true;
          playNote(294, 500);
        } else {
          Serial.println("Not matched");
          redLedOn = true;
          playNote(175, 500);
        }
        delay(1000);
        break;
      } else if (digitalRead(button3Pin) == LOW) {
        Serial.println("Button pressed: 3");
        if (userInput.equalsIgnoreCase("fear")) {
          Serial.println("Matched: " + userInput);
          greenLedOn = true;
          playNote(330, 500);
        } else {
          Serial.println("Not matched");
          redLedOn = true;
          playNote(196, 500);
        }
        delay(1000);
        break;
      } else if (digitalRead(button4Pin) == LOW) {
        Serial.println("Button pressed: 4");
        if (userInput.equalsIgnoreCase("angry")) {
          Serial.println("Matched: " + userInput);
          greenLedOn = true;
          playNote(349, 500);
        } else {
          Serial.println("Not matched");
          redLedOn = true;
          playNote(220, 500);
        }
        delay(1000);
        break;
      } else if (digitalRead(button5Pin) == LOW) {
        Serial.println("Button pressed: 5");
        if (userInput.equalsIgnoreCase("neutral")) {
          Serial.println("Matched: " + userInput);
          greenLedOn = true;
          playNote(392, 500);
        } else {
          Serial.println("Not matched");
          redLedOn = true;
          playNote(262, 500);
        }
        delay(1000);
        break;
      }
    }

    digitalWrite(greenLedPin, greenLedOn ? HIGH : LOW);
    digitalWrite(redLedPin, redLedOn ? HIGH : LOW);
    userInput = "";
  }
}
```
## Block Diagram
![image](https://github.com/vmhashim/Autism-project/blob/main/ASD/Block%20Diagram.png)

## Demo Video 


https://github.com/user-attachments/assets/6526fd61-57fb-4869-8772-e4994b786559



## Results
The system effectively recognized and responded to RFID cards representing various emotions. It consistently provided accurate feedback, lighting up the green LED and producing corresponding auditory cues when users correctly matched emotions. The system also demonstrated robustness in handling user interactions, ensuring reliable performance throughout testing phases.

## Conclusion
In conclusion, the developed system showcases reliable functionality in facilitating emotion recognition through RFID technology and user interaction. It reliably provides feedback to enhance learning and engagement, demonstrating its potential as a valuable tool for aiding individuals with Autism Spectrum Disorder (ASD) in social and emotional development. Future iterations could explore additional features such as real-time data logging for progress tracking and integration with educational software to further enhance its educational impact and usability.

This structure highlights the system's performance and potential improvements, similar to the example you provided, but tailored to the context of your project on emotion recognition using RFID technology. Adjust the specifics based on your project's outcomes and future directions.




