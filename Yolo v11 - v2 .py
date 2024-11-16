#!/usr/bin/env python
# coding: utf-8

# # Yolo v11 y arduino
# 
# En este proyecto, se explora la integración de YOLO v11, un avanzado modelo para la detección de objetos, con Arduino. La combinación de ambas tecnologías permite desarrollar sistemas que reconocen objetos y luego envían esta información a un dispositivo embebido para su utilización.

# ### Clases disponibles
# 
# Esta sección presenta una lista de clases de objetos que el modelo YOLO v11 puede identificar. 

# In[1]:


classNames = ["persona",
    "bicicleta",
    "coche",
    "moto",
    "avión",
    "autobús",
    "tren",
    "camión",
    "barco",
    "semáforo",
    "boca de incendios",
    "señal de stop",
    "parquímetro",
    "banco",
    "pájaro",
    "gato",
    "perro",
    "caballo",
    "oveja",
    "vaca",
    "elefante",
    "oso",
    "cebra",
    "jirafa",
    "mochila",
    "paraguas",
    "bolso",
    "corbata",
    "maleta",
    "frisbee",
    "esquís",
    "snowboard",
    "pelota de deporte",
    "cometa",
    "bate de béisbol",
    "guante de béisbol",
    "monopatín",
    "tabla de surf",
    "raqueta de tenis",
    "botella",
    "copa de vino",
    "taza",
    "tenedor",
    "cuchillo",
    "cuchara",
    "bol",
    "plátano",
    "manzana",
    "sándwich",
    "naranja",
    "brócoli",
    "zanahoria",
    "perrito caliente",
    "pizza",
    "donut",
    "pastel",
    "silla",
    "sofá",
    "planta en maceta",
    "cama",
    "mesa de comedor",
    "inodoro",
    "televisor",
    "portátil",
    "ratón",
    "mando a distancia",
    "teclado",
    "teléfono móvil",
    "microondas",
    "horno",
    "tostadora",
    "fregadero",
    "nevera",
    "libro",
    "reloj",
    "jarrón",
    "tijeras",
    "oso de peluche",
    "secador de pelo",
    "cepillo de dientes"]


# ### Importar librerias
# 
# Para comenzar a trabajar con el modelo YOLO v11, es necesario importar varias librerías esenciales. Se utilizaran ultralytics para el uso del modelo YOLO y cv2 de OpenCV para el procesamiento de imágenes y videos.

# In[2]:


from ultralytics import YOLO
import cv2


# ### Función para predecir
# 
# Esta sección utiliza el modelo YOLO v11 para realizar predicciones sobre imágenes o videos.Comenzamos al de cargar una imagen o video para entonces usar el modelo para detectar y clasificar los objetos presentes. 

# In[3]:


# Función para predecir objetos en una imagen usando un modelo YOLO
def predict(modelo_elegido, imagen, clases=[], conf=0.5):
    # Si se especifican clases, se predicen únicamente las clases indicadas
    if clases:
        resultados = modelo_elegido.predict(imagen, clases=clases, conf=conf)
    else:
        # Si no se especifican clases, se predicen todas con la conf dada
        resultados = modelo_elegido.predict(imagen, conf=conf)
    return resultados


# ### Dibujar cuadros delimitadores y etiquetas
# 
# Aqui vamos a utilizar el modelo de YOLO v11 cargado para llamar a la funcion para detectar los objetos y entoces se aplicaran las funciones del modelo para mostrar los resultados visualmente con etiquetas y cuadros alrededor de los objetos detectados.

# In[4]:


# Función para predecir y detectar objetos, dibujando cuadros delimitadores y etiquetas
def predict_and_detect(modelo_elegido, imagen, clases=[], conf=0.5, grosor_rectangulo=2, grosor_texto=1):
    # Realiza la predicción sobre la imagen
    resultados = predict(modelo_elegido, imagen, clases, conf=conf)
    
    for result in resultados:
        # Itera sobre cada caja detectada en el resultado
        for box in result.boxes:
            # Obtiene la clase del objeto detectado
            cls = box.cls[0]
            # Obtiene el nombre de la clase detectada
            nombre = result.names[int(cls)]
            confianza_porcentaje = box.conf[0] * 100
            print(f"{confianza_porcentaje:.2f}%")
            print(nombre) 
            
    # Dibuja un rectángulo en la imagen alrededor del objeto detectado
            cv2.rectangle(imagen, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), grosor_rectangulo)
            # Añade una etiqueta con el nombre del objeto detectado encima del rectángulo
            cv2.putText(imagen, f"{result.names[int(box.cls[0])]} ({confianza_porcentaje:.2f}%)",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), grosor_texto)
    return imagen, resultados


# ### Cargar modelo
# Aqui indicamos el modelo a usar.
# Pueden encontrar mas sobre los modelos en el Github:
# https://github.com/ultralytics/ultralytics
# 
# Ejemplos:
# - yolo11n.pt - Es el mas ligero
# - yolo11x.pt - Es el mas pesado 

# In[5]:


# Carga el modelo YOLO con el archivo "yolo11x.pt"
model = YOLO("yolo11x.pt")


# ### Predicción

# In[6]:


Nombre_de_la_imagen = "algunos-caballos-y-cientos-de-ovejas-reunidas-en-el-réttir-anual-de-islandia.jpg"
# Lee la imagen desde el archivo "zidane.jpg"
image = cv2.imread(Nombre_de_la_imagen)
# Realiza la predicción y detección sobre la imagen
result_img, _ = predict_and_detect(model, image, clases=[], conf=0.1)


# ### Resultado

# In[7]:


# Guarda la imagen resultante en un archivo
if result_img is not None:
    cv2.imwrite("result_"+Nombre_de_la_imagen , result_img)
    # Muestra la imagen resultante con las detecciones
    cv2.imshow("Resultado", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: result_img is None.")


# ## Arduino

# ### Instalar Pyserial 

# In[8]:


# instalar libreria para conectarnos por serial 
get_ipython().system('pip install pyserial')


# ### Importar librerias necesarias

# In[9]:


import serial
import time


# ### Prueba simple

# In[14]:


Puerto = "COM8"
# Configuración del puerto serial (ajusta el puerto a tu configuración)
arduino = serial.Serial(port=Puerto, baudrate=921600, timeout=1)

time.sleep(2)  # Esperar a que Arduino esté listo

def enviar_comando(comando):
    arduino.write(f"{comando}\n".encode())  # Enviar comando
    time.sleep(0.5)  # Esperar por la respuesta
    respuesta = arduino.readline().decode().strip()  # Leer respuesta
    print(f"Respuesta de Arduino: {respuesta}")


# In[11]:


# Enviar comandos
enviar_comando("on")   # Encender LED
time.sleep(2)           # Esperar 2 segundos
enviar_comando("off")  # Apagar LED
time.sleep(2)
enviar_comando("gato")   # Encender LED
time.sleep(2)
enviar_comando("dog")   # Encender LED

# Cerrar la conexión al final
arduino.close()


# ### Funcion de deteccion y envio de informacion

# In[7]:


# Función para predecir y detectar objetos, dibujando cuadros delimitadores y etiquetas
def predict_and_detect(modelo_elegido, imagen, clases=[], conf=0.5, grosor_rectangulo=2, grosor_texto=1):
    # Realiza la predicción sobre la imagen
    resultados = predict(modelo_elegido, imagen, clases, conf=conf)
    
    for result in resultados:
        # Itera sobre cada caja detectada en el resultado
        for box in result.boxes:
            # Obtiene la clase del objeto detectado
            cls = box.cls[0]
            # Obtiene el nombre de la clase detectada
            nombre = classNames[int(cls)]
            confianza_porcentaje = box.conf[0] * 100
            print(f"{confianza_porcentaje:.2f}%")
            print(nombre) 

            if nombre == "gato":
                enviar_comando("gato")   # Encender LED
            
    # Dibuja un rectángulo en la imagen alrededor del objeto detectado
            cv2.rectangle(imagen, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), grosor_rectangulo)
            # Añade una etiqueta con el nombre del objeto detectado encima del rectángulo
            cv2.putText(imagen, f"{result.names[int(box.cls[0])]} ({confianza_porcentaje:.2f}%)",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), grosor_texto)
    return imagen, resultados


# In[19]:


Nombre_de_la_imagen = "-703-2048x1070-0.jpg"
# Lee la imagen desde el archivo "zidane.jpg"
image = cv2.imread(Nombre_de_la_imagen)
# Realiza la predicción y detección sobre la imagen
result_img, _ = predict_and_detect(model, image, clases=[], conf=0.1)


# In[ ]:


# Guarda la imagen resultante en un archivo
if result_img is not None:
    cv2.imwrite("result_"+Nombre_de_la_imagen , result_img)
    # Muestra la imagen resultante con las detecciones
    cv2.imshow("Resultado", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: result_img is None.")


# In[9]:


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Perform object detection using the existing function
    frame, _ = predict_and_detect(model, frame)

    # Display the resulting frame
    cv2.imshow("YOLO v11 Webcam Detection", frame)

    # Press 'q' to quit the webcam stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# In[15]:


import cv2
from ultralytics import YOLO

# Load YOLOv11 model (use your model path here)
model = YOLO("yolo11n.pt")  # Update with your model's actual path

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

        # Print detection details
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)  # Convert tensor to int
                confidence = float(box.conf)  # Convert tensor to float
                bbox = box.xyxy.tolist()  # Convert tensor to list
                name = result.names[int(class_id)]
                # Print detection information
                print("Name: " + name + "")
                print(f"Class: {class_id}, Confidence: {confidence:.2f}, BBox: {bbox}")

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


# In[ ]:




