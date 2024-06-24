# Importamos las librerias
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO("bestColab.pt")

# Realizar VideoCaptura
cap = cv2.VideoCapture(1)

# Diccionario para almacenar las instancias detectadas
instance_crops_zoomed = {}

# Variable de control para detener la búsqueda
all_instances_found = False

# Bucle
while not all_instances_found:
    # Leer nuestros fotogramas
    ret, frame = cap.read()

    # Leemos resultados
    resultados = model.predict(frame, imgsz=640, conf=0.45)

    # Recortar y almacenar cada instancia detectada
    for i, box in enumerate(resultados[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        instance_crop = frame[y1:y2, x1:x2]

        # Zoom de 300%
        zoom_factor = 3.0
        instance_crop_zoomed = cv2.resize(instance_crop, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

        # Obtener la etiqueta de clase de la instancia detectada
        class_id = int(box.cls[0])
        class_names = model.names
        class_name = class_names[class_id]

        # Almacenar la imagen recortada con el nombre de la clase
        instance_crops_zoomed[class_name] = instance_crop_zoomed

    # Mostrar cada instancia en su propia ventana con alta calidad
    for class_name, instance_crop_zoomed in instance_crops_zoomed.items():
        # Ajustar la ventana para alta calidad
        cv2.namedWindow(class_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(class_name, instance_crop_zoomed.shape[1], instance_crop_zoomed.shape[0])

        # Mostrar la imagen en la ventana
        cv2.imshow(class_name, instance_crop_zoomed)

    # Verificar si todas las anotaciones han sido encontradas
    if len(instance_crops_zoomed) == 6:
        all_instances_found = True

    # Esperar a que el usuario cierre las ventanas o presione 'q' para salir
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

# Liberar recursos y mantener las ventanas abiertas
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
