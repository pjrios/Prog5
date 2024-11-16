#!/usr/bin/env python
# coding: utf-8

# # Yolo v11 y arduino
# 
# En este proyecto, se explora la integración de YOLO v11, un avanzado modelo para la detección de objetos, con Arduino. La combinación de ambas tecnologías permite desarrollar sistemas que reconocen objetos **en tiempo real** y luego envían esta información a un dispositivo embebido para su utilización.

# In[17]:


from ultralytics import YOLO
import serial
import time
import cv2


# ## Importar modelo

# In[18]:


# Load YOLOv11 model
model = YOLO("yolo11n.pt")  # Update with your model's actual path


# ## Definir funciones y variables

# In[19]:


Puerto = "COM9"
# Configuración del puerto serial (ajusta el puerto a tu configuración)
arduino = serial.Serial(port=Puerto, baudrate=9600, timeout=1)

time.sleep(2)  # Esperar a que Arduino esté listo

def enviar_comando(comando):
    arduino.write(f"{comando}\n".encode())  # Enviar comando
    time.sleep(0.5)  # Esperar por la respuesta
    respuesta = arduino.readline().decode().strip()  # Leer respuesta
    print(f"Respuesta de Arduino: {respuesta}")

# Function to process detections
def process_detections(results):
    # Iterate over each result in the results list
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)  # Convert tensor to int
            confidence = float(box.conf)  # Convert tensor to float
            bbox = box.xyxy.tolist()  # Convert tensor to list
            name = result.names[class_id]

            # Print detection information
            print(f"Name: {name}")
            #print(f"Class: {class_id}, Confidence: {confidence:.2f}, BBox: {bbox}")

            # Send command if the detected object is "cat"
            #if name == "person":
            enviar_comando(name)  # Example: Send command to turn on LED


# ## Main program

# In[20]:


# Open the webcam
cap = cv2.VideoCapture(0)  # '0' selects the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    cap.release()
    exit()

# Read frames in a loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Run YOLO on the frame
        results = model(frame)  # Perform inference

        # Process detection results
        process_detections(results)

        # Draw the results on the frame
        annotated_frame = results[0].plot()  # Annotate the frame with detections

        # Display the annotated frame
        cv2.imshow("YOLOv11 Webcam", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    # Cerrar la conexión al final
    arduino.close()


# ```c
# // Código Arduino para controlar una luz basado en comandos recibidos por serial
# // Pin conectado al LED o luz 
# const int lightPin = 13; 
# // Variable para almacenar el último momento en que se detectó una persona
# unsigned long lastPersonDetectedTime = 0; 
# // Tiempo límite de 1 segundo (1000 ms)
# const unsigned long timeout = 10000; 
# 
# void setup() {
#   // Inicializa la comunicación serial a 921600 baudios
#   Serial.begin(9600); 
#   // Configura el pin del LED como salida
#   pinMode(lightPin, OUTPUT); 
# }
# 
# void loop() {
#   // Verifica si hay datos disponibles para leer desde el puerto serial
#   if (Serial.available() > 0) {
#     // Lee el comando entrante hasta encontrar un salto de línea
#     String command = Serial.readStringUntil('\n'); 
#     // Elimina cualquier espacio en blanco o caracteres de nueva línea al final del comando
#     command.trim(); 
#     if (command == "person") { 
#       // Si el comando recibido es "person"
#       digitalWrite(lightPin, HIGH); // Enciende la luz
#       lastPersonDetectedTime = millis(); // Actualiza el tiempo de la última detección
#       Serial.println("On");
#     }
#   }
#   // Apaga la luz si no se recibe el comando "person" durante 1 segundo
#   if (millis() - lastPersonDetectedTime > timeout) {
#     digitalWrite(lightPin, LOW); // Apaga la luz
#     Serial.println("Off");
#   }
# }
# ```

# In[ ]:




